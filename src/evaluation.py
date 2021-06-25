from tqdm import tqdm
from src.model import load_model
from src.dataset import get_dataloader
import numpy as np
import random
import torch
import os

OOM_RETRY_LIMIT = 10


def evaluate(dataloader, model):
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    model.eval()

    loss_sum = 0
    oom_counter = 0
    n_sample = 0

    for batch in tqdm(dataloader, desc="Iteration"):
        with torch.no_grad():
            try:
                # load data and compute loss
                indexes, anchor_sample, positive_sample = batch
                anchor_sample, positive_sample = (
                    anchor_sample.to(device),
                    positive_sample.to(device),
                )
                loss_sum += model(anchor_sample, positive_sample).item()

                batch_size = len(indexes)

                # compute n_sample
                n_sample += batch_size

            except RuntimeError as e:
                if not "CUDA out of memory" in str(e):
                    raise
                if oom_counter >= OOM_RETRY_LIMIT:
                    oom_counter = 0
                    break
                oom_counter += 1
                torch.cuda.empty_cache()

    model.train()
    torch.cuda.empty_cache()

    loss_avg = loss_sum / n_sample
    return loss_avg

def predict(args):
    assert not args.ckpt is None
    _, model, _, _ = load_model(args.ckpt)
    model = model.to(args.device)
    fever_loader = get_dataloader(args, train=False)
    with torch.no_grad():
        for batch in tqdm(fever_loader, desc="Iteration"):
            claim = [data['claim'] for data in  batch]
            # claim = [data['evidences'][0]['document'][1] for data in  batch]
            evdn = [data['evidences'][0]['document'][1] for data in  batch]
            print(claim[0])
            print(evdn[0])
            
            clm_vec = model.ctx2vec(claim, args.device)
            evdn_vec = model.ctx2vec(evdn, args.device)
            print((clm_vec * evdn_vec).sum(dim=-1).mean())
            random.shuffle(evdn)
            evdn_vec = model.ctx2vec(evdn, args.device)
            print((clm_vec * evdn_vec).sum(dim=-1).mean())
            print('----------------------------------')