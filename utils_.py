from dataset import DummyDataset
from backbone import LSTM
from contrastor.contrastive_module import RetrievalModelWrapper
from contrastor.contrastive_loss import NCELoss
import torch


def build_model(args, config):
    print(f'[Runner] - Building contrastive model')
    loss_config = config['loss'][f'{args.loss}']
    loss_config['dim'] = config['model']['LSTM']['output_size']

    if args.loss in ['InfoNCE', 'ProtoNCE', 'HProtoNCE']:
        criterion = NCELoss(loss_config)

    bk_model = eval(args.model)(config)
    use_LSTM = isinstance(bk_model, LSTM)

    model = RetrievalModelWrapper(
        bk_model, criterion, loss_config, use_LSTM=use_LSTM)
    return model


def get_dataloader(config, args, train=True, bsz=None):
    if bsz is None:
        bsz = config['train']['batch_size'] if train else config['eval']['batch_size']
    n_jobs = config['train']['n_jobs'] if train else config['eval']['n_jobs']

    dataset = DummyDataset()

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=train,
        num_workers=n_jobs,
        drop_last=train
        # collate_fn=dataset.collate_fn
    )


def get_optimizer(args, config, model):
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=float(config['optimizer']
                                             ['SGD']['learning_rate']),
                                    momentum=float(
                                        config['optimizer']['SGD']['momentum']),
                                    weight_decay=float(config['optimizer']['SGD']['weight_decay']))
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=float(config['optimizer']
                                              ['Adam']['learning_rate']),
                                     betas=tuple(config['optimizer']['Adam']['betas']))
    return optimizer


def save_model(model, optimizer, args, config, current_step):
    path = f'{args.ckptdir}/{args.loss}_{args.model}_{current_step}.pth'
    all_states = {
        'Model': model.state_dict(),
        'Optimizer': optimizer.state_dict(),
        'Current_step': current_step,
        'Config': config,
        'Args': args
    }
    torch.save(all_states, path)


def load_model(path):
    ckpt = torch.load(path, map_location='cpu')
    args, config = ckpt['Args'], ckpt['Config']

    model = build_model(args, config)
    print(f'[Runner] - Loading model parameters')
    model.load_state_dict(ckpt['Model'])

    optimizer = get_optimizer(args, config, model)
    optimizer.load_state_dict(ckpt['Optimizer'])

    current_step = ckpt['Current_step']
    return args, config, model, optimizer, current_step


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
