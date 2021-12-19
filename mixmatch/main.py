#%%
import argparse
import os

import numpy as np
import tensorflow as tf
import tqdm
import yaml

from preprocess import fetch_dataset
#%%
config = tf.compat.v1.ConfigProto()
'''
GPU 메모리를 전부 할당하지 않고, 아주 적은 비율만 할당되어 시작됨
프로세스의 메모리 수요에 따라 자동적으로 증가
but
GPU 메모리를 처음부터 전체 비율을 사용하지 않음
'''
config.gpu_options.allow_growth = True

'''
분산 학습 설정
'''
strategy = tf.distribute.MirroredStrategy()
# session = tf.compat.v1.InteractiveSession(config=config)
#%%
# def get_args():
parser = argparse.ArgumentParser('parameters')

parser.add_argument('--seed', type=int, default=None, 
                    help='seed for repeatable results')
parser.add_argument('--dataset', type=str, default='svhn',
                    help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra)')

parser.add_argument('--epochs', type=int, default=1024, 
                    help='number of epochs, (default: 1024)')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='examples per batch (default: 64)')
parser.add_argument('--learning_rate', type=float, default=1e-2, 
                    help='learning_rate, (default: 0.01)')

parser.add_argument('--labeled_examples', type=int, default=4000, 
                    help='number labeled examples (default: 4000')
parser.add_argument('--validation_examples', type=int, default=5000, 
                    help='number validation examples (default: 5000')
parser.add_argument('--val_iteration', type=int, default=1024, 
                    help='number of iterations before validation (default: 1024)')
parser.add_argument('--T', type=float, default=0.5, 
                    help='temperature sharpening ratio (default: 0.5)')
parser.add_argument('--K', type=int, default=2, 
                    help='number of rounds of augmentation (default: 2)')
parser.add_argument('--alpha', type=float, default=0.75,
                    help='param for sampling from Beta distribution (default: 0.75)')
parser.add_argument('--lambda_u', type=int, default=100, 
                    help='multiplier for unlabeled loss (default: 100)')
parser.add_argument('--rampup_length', type=int, default=16,
                    help='rampup length for unlabelled loss multiplier (default: 16)')
parser.add_argument('--weight_decay', type=float, default=0.02, 
                    help='decay rate for model vars (default: 0.02)')
parser.add_argument('--ema_decay', type=float, default=0.999, 
                    help='ema decay for ema model vars (default: 0.999)')

parser.add_argument('--config_path', type=str, default=None, 
                    help='path to yaml config file, overwrites args')
parser.add_argument('--tensorboard', action='store_true', 
                    help='enable tensorboard visualization')
parser.add_argument('--resume', action='store_true', 
                    help='whether to restore from previous training runs')

    # return parser.parse_args()
#%%
def load_config(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, args['config_path'])    
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args
#%%
# def main():

'''argparse to dictionary'''
# args = vars(get_args())
'''argparse debugging'''
args = vars(parser.parse_args(args=['--epochs', '150', '--T', '0.1']))

dir_path = os.path.dirname(os.path.realpath(__file__))
args['config_path'] = 'configs/cifar10_4000.yaml'
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

start_epoch = 0
log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'
ckpt_dir = f'{log_path}/checkpoints'
#%%
datasetL, datasetU, val_dataset, test_dataset, num_classses = fetch_dataset(args, log_path)
#%%


#%%