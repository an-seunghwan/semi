#%%
'''
20211223-210235
'''
#%%
import argparse
import os

# os.chdir(r'D:\semi\sl\unconditional') # main directory (repository)
os.chdir('/home1/prof/jeon/an/semi/sl/unconditional') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm
import yaml
import io
import matplotlib.pyplot as plt

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import VAE
# from criterion import ELBO_criterion
from mixup import augment
#%%
config = tf.compat.v1.ConfigProto()
'''
GPU 메모리를 전부 할당하지 않고, 아주 적은 비율만 할당되어 시작됨
프로세스의 메모리 수요에 따라 자동적으로 증가
but
GPU 메모리를 처음부터 전체 비율을 사용하지 않음
'''
config.gpu_options.allow_growth = True

# '''
# 분산 학습 설정
# '''
# strategy = tf.distribute.MirroredStrategy()
# session = tf.compat.v1.InteractiveSession(config=config)
#%%
# import ast
# def arg_as_list(s):
#     v = ast.literal_eval(s)
#     if type(v) is not list:
#         raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
#     return v
#%%
def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results (ex. generating color MNIST)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra, cmnist)')
    # parser.add_argument("--image_size", default=32, type=int,
    #                     metavar='Image Size', help='the size of height or width for image')
    # parser.add_argument("--channel", default=3, type=int,
    #                     metavar='Channel size', help='the size of image channel')
    parser.add_argument('--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=600, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=50, type=int,
                        metavar='N', help='reconstruct frequency (default: 50)')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters (Encoder and Decoder)'''
    # parser.add_argument('--net-name', default="wideresnet-28-2", type=str, help="the name for network to use")
    parser.add_argument('--depth', type=int, default=28, 
                        help='depth for WideResnet (default: 28)')
    parser.add_argument('--width', type=int, default=2, 
                        help='widen factor for WideResnet (default: 2)')
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='slope parameter for LeakyReLU (default: 0.1)')
    parser.add_argument('-dr', '--drop_rate', default=0, type=float, 
                        help='drop rate for the network')
    parser.add_argument("--br", "--bce_reconstruction", action='store_true', 
                        help='Do BCE Reconstruction')
    parser.add_argument('--x_sigma', default=1, type=float,
                        help="The standard variance for reconstructed images, work as regularization")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=128, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    # parser.add_argument('--cmi', "--continuous_mutual_info", default=0, type=float,
    #                     help='The mutual information bounding between x and the continuous variable z')
    # parser.add_argument('--dmi', "--discrete_mutual_info", default=0, type=float,
    #                     help='The mutual information bounding between x and the discrete variable z')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--lambda1', default=5., type=float,
                        help="adjust classification loss weight")
    parser.add_argument('--lambda2', default=10., type=float,
                        help="adjust mutual information loss weight")
    # parser.add_argument('--ewm', '--elbo-weight-max', default=1e-3, type=float, 
    #                     metavar='weight for elbo loss part')
    # parser.add_argument('--aew', '--adjust-elbo-weight', default=400, type=int,
    #                     metavar="the epoch to adjust elbo weight to max")
    # parser.add_argument('--wrd', default=1, type=float,
    #                     help="the max weight for the optimal transport estimation of discrete variable c")
    # parser.add_argument('--wmf', '--weight-modify-factor', default=0.4, type=float,
    #                     help="weight  will get wrz at amf * epochs")
    # parser.add_argument('--pwm', '--posterior-weight-max', default=1, type=float,
    #                     help="the max value for posterior weight")
    # parser.add_argument('--apw', '--adjust-posterior-weight', default=200, type=float,
    #                     help="adjust posterior weight")

    '''Optimizer Parameters (Encoder and Decoder)'''
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    # parser.add_argument('--beta1', default=0.9, type=float, metavar='Beta1 In ADAM and SGD',
    #                     help='beta1 for adam as well as momentum for SGD')
    # parser.add_argument('--adjust_lr', default=[400, 500, 550], type=arg_as_list,
    #                     help="The milestone list for adjust learning rate")
    # parser.add_argument('--weight_decay', default=5e-4, type=float)

    # '''Optimizer Transport Estimation Parameters'''
    # parser.add_argument('--epsilon', default=0.1, type=float,
    #                     help="the label smoothing epsilon for labeled data")
    # # parser.add_argument('--om', action='store_true', help="the optimal match for unlabeled data mixup")
    
    '''Normalizing Flow Model Parameters'''
    parser.add_argument('--z_mask', default='checkerboard', type=str,
                        help='mask type of continuous latent for Real NVP (e.g. checkerboard or half)')
    parser.add_argument('--c_mask', default='half', type=str,
                        help='mask type of discrete latent for Real NVP (e.g. checkerboard or half)')
    parser.add_argument('--z_emb', default=256, type=int,
                        help='embedding dimension of continuous latent for coupling layer')
    parser.add_argument('--c_emb', default=256, type=int,
                        help='embedding dimension of discrete latent for coupling layer')
    parser.add_argument('--K1', default=8, type=int,
                        help='number of coupling layers in Real NVP (continous latent)')
    parser.add_argument('--K2', default=8, type=int,
                        help='number of coupling layers in Real NVP (discrete latent)')
    parser.add_argument('--coupling_MLP_num', default=4, type=int,
                        help='number of dense layers in single coupling layer')
    
    '''Normalizing Flow Optimizer Parameters'''
    parser.add_argument('--lr_nf', '--learning-rate-nf', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate for normalizing flow')
    parser.add_argument('--reg', default=0.01, type=float,
                        help='L2 regularization parameter for dense layers in Real NVP')
    parser.add_argument('--decay_steps', default=1, type=int,
                        help='decay steps for exponential decay schedule')
    parser.add_argument('--decay_rate', default=0.95, type=float,
                        help='decay rate for exponential decay schedule')
    parser.add_argument('--gradclip', default=1., type=float,
                        help='gradclip value')

    '''Configuration'''
    parser.add_argument('--config_path', type=str, default=None, 
                        help='path to yaml config file, overwrites args')

    return parser.parse_args()
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
def generate_and_save_images(model, image, num_classes):
    z = model.ae.z_encode(image, training=False)
    
    buf = io.BytesIO()
    figure = plt.figure(figsize=(10, 2))
    plt.subplot(1, num_classes+1, 1)
    plt.imshow((image[0] + 1) / 2)
    plt.title('original')
    plt.axis('off')
    for i in range(num_classes):
        label = np.zeros((z.shape[0], num_classes))
        label[:, i] = 1
        xhat = model.ae.decode(z, label, training=False)
        plt.subplot(1, num_classes+1, i+2)
        plt.imshow((xhat[0] + 1) / 2)
        plt.title('{}'.format(i))
        plt.axis('off')
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
#%%
def main():
    '''argparse to dictionary'''
    args = vars(get_args())
    # '''argparse debugging'''
    # args = vars(parser.parse_args(args=['--config_path', 'configs/cmnist.yaml']))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}'

    dataset, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    
    model = VAE(args, num_classes)
    model.build(input_shape=(None, 32, 32, 3))
    # model.summary()
#%%