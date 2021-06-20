import yaml
import random
import os
import numpy as np
import torch
import argparse
from src.train import train
import _pickle as pk
from src.dataset import get_dataloader
from transformers import BertTokenizer, BertModel, AutoTokenizer
from src.train import bert_extractor

import time

def get_args():
    # contruct parser object
    parser = argparse.ArgumentParser(description='Argument Parser.')

    parser.add_argument('--config', type=str, help='Path to experiment configuration.', default='config.yaml')
    parser.add_argument('--log', action='store_true', default=False,
                        help='Recording loss and metric scores.')
    parser.add_argument('--logdir', default='log', type=str, help='Directory for logging.')
    parser.add_argument('--ckptdir', default='ckpt', type=str,
                        help='Path to store checkpoint result, if empty then default is used.')

    # Options
    parser.add_argument('--seed', default=1337, type=int,
                        help='Random seed for reproducable results.')
    parser.add_argument('--gpu', default='0', type=str, help='Assigning GPU id. (-1: use CPU)')
    parser.add_argument('--ckpt', type=str, help="Path to load target pretrain model")

    # contrastive learning argument
    parser.add_argument('--model', default='LSTM', type=str, choices=['LSTM'], help="Selection of module type")
    parser.add_argument('--loss', default='InfoNCE', type=str, choices=['InfoNCE', 'ProtoNCE', 'HProtoNCE'],
                             help="Selection of contrastive loss type")
    parser.add_argument('--opt', default='adam', type=str, choices=['adam', 'sgd'],
                             help="Selection of optimizer type")
    
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
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    args.device = torch.device('cpu') if int(args.gpu.split(',')[0]) < 0 else torch.device('cuda:' + (args.gpu))
    
    with open(args.config['dataset']['docs_sentence'], 'rb') as f:
        data = pk.load(f)
        
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.cuda()
    bert_model.eval()

    train(data, bert_model, bert_tokenizer, args)
