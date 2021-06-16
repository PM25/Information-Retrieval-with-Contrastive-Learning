from parse import get_args
import yaml
import random
import os
import numpy as np
import torch


def main():
    # parse
    args = get_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load configure
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    from train import train
    train(args, config)

if __name__ == "__main__":
    main()
