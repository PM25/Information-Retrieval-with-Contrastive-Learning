from torch.utils.tensorboard import SummaryWriter
from utils_ import build_model, get_optimizer, save_model, load_model, optimizer_to, get_dataloader
from contrastor.utils import run_kmeans, run_hierarchical_clustering
from evaluation import evaluate
from tqdm import tqdm

import numpy as np
import torch
import os
import math
import random
import faiss
import shutil
import time


def adjust_learning_rate(optimizer, steps, config):
    """Decay the learning rate based on schedule"""
    lr = float(config['optimizer']['SGD']['learning_rate'])
    lr *= 0.5 * (1. + math.cos(math.pi * steps /
                               config['train']['total_steps']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, config):
    # set cuda device
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set initialization
    init_step = 0
    if args.ckpt is None:
        model = build_model(args, config)
        optimizer = get_optimizer(args, config, model)
    else:
        _, _, model, optimizer, init_step = load_model(args.ckpt)
        
    model = model.to(device)
    optimizer_to(optimizer, device)

    # set batch size
    assert config['train']['acml_batch_size'] % config['train']['batch_size'] == 0
    batch_size = 0
    acml_batch_size = config['train']['acml_batch_size']
    
    # todo
    train_loader = get_dataloader(config, args, train=True)
    dev_loader = get_dataloader(config, args, train=False)
    if args.loss in ['ProtoNCE', 'HProtoNCE']:
        feat_loader = get_dataloader(config, args, train=False)
    cluster_result = None

    # build logger directory
    args.logdir = f'{args.logdir}/{args.loss}_{args.model}'
    if os.path.isdir(args.logdir):
        shutil.rmtree(args.logdir)
    os.makedirs(args.logdir)
    log = SummaryWriter(args.logdir)

    # build ckpt directory
    os.makedirs(args.ckptdir, exist_ok=True)

    # set steps
    loss_record = []
    loss_sum = 0
    step_sum = init_step
    total_steps = config['train']['total_steps']

    print(f'[Runner] - Start training')
    pbar = tqdm(initial=init_step, total=total_steps, dynamic_ncols=True)

    while step_sum < total_steps:
        epoch = step_sum // len(train_loader)

        # scheduler process
        if args.opt == 'sgd':
            adjust_learning_rate(optimizer, step_sum, config)

        for batch in train_loader:
            try:
                # extract all noise embeddings
                if args.loss == 'ProtoNCE' and batch_size == 0 \
                        and step_sum >= config['loss']['ProtoNCE']['cluster_start_steps'] \
                        and step_sum % config['loss']['ProtoNCE']['cluster']['update_steps'] == 0:
                    cluster_result = run_kmeans(
                        config['loss']['ProtoNCE'], feat_loader, model, device)
                if args.loss == 'HProtoNCE' and batch_size == 0 \
                        and step_sum >= config['loss']['HProtoNCE']['cluster_start_steps'] \
                        and step_sum % config['loss']['HProtoNCE']['cluster']['update_steps'] == 0:
                    cluster_result = run_hierarchical_clustering(
                        config['loss']['HProtoNCE'], feat_loader, model, device)

                # start using queue
                if model.use_queue and step_sum >= config['loss'][args.loss]['queue_start_steps'] and not model.add_queue_to_loss:
                    model.add_queue_to_loss = True

                # load data
                indexes, anchor_sample, positive_sample = batch
                anchor_sample, positive_sample = anchor_sample.to(device), positive_sample.to(device)
                batch_size += len(indexes)

                # process forward and backward
                loss = model(anchor_sample, positive_sample, cluster_result,
                                indexes) / acml_batch_size
                loss.backward()
                loss_sum += loss.item()

           
                if batch_size == acml_batch_size or len(indexes) != config['train']['batch_size']:
                    # gradient clipping
                    if args.model == 'LSTM':
                        down_paras = list(model.parameters())
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            down_paras, config['optimizer']['gradient_clipping'])
                        # update parameters
                        if math.isnan(grad_norm) or math.isinf(grad_norm):
                            print(
                                '[Runner] - Error : grad norm is nan/inf at step {step_sum}')
                    optimizer.step()
                    if model.use_momentum:
                        model._momentum_update_key_encoder()

                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    loss_record.append(loss_sum)
                    pbar.update(1)
                    step_sum += 1
                    batch_size = 0
                    loss_sum = 0

                # log training recording
                if (step_sum != init_step and step_sum % int(config['train']['log_step']) == 0) and batch_size == 0:
                    loss_avg = np.mean(loss_record)
                    loss_record = []
                    log.add_scalar('train_loss', loss_avg, step_sum)
                    log.add_scalar('grad_norm', grad_norm, step_sum)

                # evaluate and save the best
                if (step_sum != init_step and step_sum % int(config['train']['eval_step']) == 0) and batch_size == 0:
                    print(f'[Runner] - Evaluating on development set')
                    loss = evaluate(args, config, dev_loader, model)
                    log.add_scalar('dev_loss', loss, step_sum)
                    pbar.set_description(
                        'Train_Loss %.5f | Valid_Loss %.5f' % (loss_avg, loss))
                    save_model(model, optimizer, args, config, step_sum)

            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise
                print('[Runner] - CUDA out of memory at step: ', step_sum)
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            if step_sum >= total_steps:
                break

    pbar.close()
    log.close()
