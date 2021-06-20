import torch
import torch.nn as nn
from src.contrastor.contrastive_module import RetrievalModelWrapper
from src.contrastor.contrastive_loss import NCELoss


class LSTM(nn.Module):
    def __init__(self, config, **kwargs):
        super(LSTM, self).__init__()
        input_size = config["model"]["LSTM"]["input_size"]
        hidden_size = config["model"]["LSTM"]["hidden_size"]
        output_size = config["model"]["LSTM"]["output_size"]
        bidirectional = config["model"]["LSTM"]["bidirectional"]
        act = config["model"]["LSTM"]["activation"]

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=config["model"]["LSTM"]["num_layers"],
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.scaling_layer = nn.Sequential(
            nn.Linear(max(1, int(bidirectional) * 2) * hidden_size, output_size),
            eval(f"nn.{act}()"),
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name or "scaling_layer.0.weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

    def forward(self, features, **kwargs):
        predicted, _ = self.lstm(features)
        predicted = self.scaling_layer(predicted)
        return predicted
    

def get_optimizer(args, model):
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=float(args.config['optimizer']['SGD']['learning_rate']),
                                    momentum=float(args.config['optimizer']['SGD']['momentum']),
                                    weight_decay=float(args.config['optimizer']['SGD']['weight_decay']))
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=float(args.config['optimizer']['Adam']['learning_rate']),
                                     betas=tuple(args.config['optimizer']['Adam']['betas']))
    return optimizer


def build_model(args):
    print(f"[Runner] - Building contrastive model")
    loss_config = args.config["loss"][f"{args.loss}"]
    loss_config["dim"] = args.config["model"]["LSTM"]["output_size"]

    if args.loss in ["InfoNCE", "ProtoNCE", "HProtoNCE"]:
        criterion = NCELoss(loss_config)

    bk_model = eval(args.model)(args.config)
    use_LSTM = isinstance(bk_model, LSTM)

    model = RetrievalModelWrapper(bk_model, criterion, loss_config, use_LSTM=use_LSTM)
    return model


def save_model(model, optimizer, args, current_step):
    path = f"{args.ckptdir}/{args.loss}_{args.model}_{current_step}.pth"
    all_states = {
        "Model": model.state_dict(),
        "Optimizer": optimizer.state_dict(),
        "Current_step": current_step,
        "Args": args,
    }
    torch.save(all_states, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt["Args"]

    model = build_model(args)
    print(f"[Runner] - Loading model parameters")
    model.load_state_dict(ckpt["Model"])

    optimizer = get_optimizer(args, model)
    optimizer.load_state_dict(ckpt["Optimizer"])

    current_step = ckpt["Current_step"]
    return args, model, optimizer, current_step