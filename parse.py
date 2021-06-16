import argparse
import os

def assertion_check(args):
    if args.log:
        assert args.logdir is not None    
    assert args.model is not None

    if args.config is None:
        args.config = 'config/retv.yaml'


def get_args():
    # contruct parser object
    parser = argparse.ArgumentParser(
        description='Argument Parser.')

    parser.add_argument('--config', type=str, help='Path to experiment configuration.')
    parser.add_argument('--log', action='store_true',
                        help='Recording loss and metric scores.')
    parser.add_argument('--logdir', default='log', type=str,
                        help='Directory for logging.', required=False)
    parser.add_argument('--ckptdir', default='ckpt', type=str,
                        help='Path to store checkpoint result, if empty then default is used.', required=False)

    # Options
    parser.add_argument('--seed', default=1337, type=int,
                        help='Random seed for reproducable results.', required=False)
    parser.add_argument('--cpu', action='store_true',
                        help='Disable GPU training.')
    parser.add_argument('--gpu', default='2', type=int,
                        help='Assigning GPU id.')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Enable Multi-GPU training.')
    parser.add_argument(
        '--ckpt', type=str, help="Path to load target pretrain model")

    # contrastive learning argument
    parser.add_argument('--model', default='LSTM', type=str, choices=['LSTM'],
                             help="Selection of module type")
    parser.add_argument('--loss', default='InfoNCE', type=str, choices=['InfoNCE', 'ProtoNCE', 'HProtoNCE'],
                             help="Selection of contrastive loss type")
    parser.add_argument('--opt', default='adam', type=str, choices=['adam', 'sgd'],
                             help="Selection of optimizer type")
    
    # get parsing results
    args = parser.parse_args()
    assertion_check(args)
    return args
