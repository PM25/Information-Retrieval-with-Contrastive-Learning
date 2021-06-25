import os
import yaml
import time
import torch
import random
import argparse
import numpy as np
import _pickle as pk

from src.train import train
from src.dataset import get_dataloader


def get_args():
    # contruct parser object
    parser = argparse.ArgumentParser(description="Argument Parser.")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration.",
        default="config.yaml",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Recording loss and metric scores.",
    )
    parser.add_argument(
        "--logdir", default="log", type=str, help="Directory for logging."
    )
    parser.add_argument(
        "--data", default="doc", type=str, help="Directory for training/testing data."
    )
    parser.add_argument(
        "--ckptdir",
        default="ckpt",
        type=str,
        help="Path to store checkpoint result, if empty then default is used.",
    )

    # Options
    parser.add_argument(
        "--seed", default=1337, type=int, help="Random seed for reproducable results."
    )
    parser.add_argument(
        "--gpu", default="0", type=str, help="Assigning GPU id. (-1: use CPU)"
    )
    parser.add_argument("--ckpt", type=str, help="Path to load target pretrain model")

    # contrastive learning argument
    parser.add_argument(
        "--model",
        default="LSTM",
        type=str,
        choices=["LSTM"],
        help="Selection of module type",
    )
    parser.add_argument(
        "--loss",
        default="InfoNCE",
        type=str,
        choices=["InfoNCE", "ProtoNCE", "HProtoNCE"],
        help="Selection of contrastive loss type",
    )
    parser.add_argument(
        "--opt",
        default="adam",
        type=str,
        choices=["adam", "sgd"],
        help="Selection of optimizer type",
    )
    parser.add_argument(
        "--sample",
        default="uniform",
        type=str,
        choices=["uniform", "tf_idf"],
        help="Sampling methods",
    )

    # get parsing results
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # parse
    args = get_args()

    # set random seed
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load configure
    args.config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    args.device = (
        torch.device("cpu")
        if int(args.gpu.split(",")[0]) < 0
        else torch.device("cuda:" + (args.gpu))
    )

    if args.data == 'doc':
        # start training
        train(args)
    elif args.data == 'fever':
        from src.evaluation import predict
        predict(args)