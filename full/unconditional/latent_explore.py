#%%
import argparse
import os

os.chdir(r'D:\semi\sl\unconditional') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/sl/unconditional') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds
import tqdm
import yaml
from PIL import Image
import io
import matplotlib.pyplot as plt

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import VAE
# from criterion import ELBO_criterion
from mixup import augment
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

    return parser
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
args = vars(get_args().parse_args(args=['--config_path', 'configs/cmnist.yaml']))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}'

dataset, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)

model_path = log_path + '/20211228-203211'
model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
model = VAE(args, num_classes)
model.build(input_shape=(None, 32, 32, 3))
model.load_weights(model_path + '/' + model_name)
model.summary()
#%%
x = []
y = []
for example in dataset:
    x.append(example[0])
    y.append(example[1])
    if len(x) == 5000: break
x = tf.cast(np.array(x), tf.float32)
y = tf.cast(np.array(y), tf.float32)
#%%
'''interpolation: c'''
lam = 0.2
tf.random.set_seed(1)
x_ = tf.gather(x, tf.random.shuffle(tf.range(tf.shape(x)[0])))
x_inter = lam * x + (1. - lam) * x_

# interpolation on x
prob_x_inter = tf.nn.softmax(model.ae.c_encode(x_inter))

# interpolation on c
prob_c_inter = tf.nn.softmax(lam * model.ae.c_encode(x) + (1. - lam) * model.ae.c_encode(x_))

# interpolation on softmax
prob1 = tf.nn.softmax(model.ae.c_encode(x))
prob2 = tf.nn.softmax(model.ae.c_encode(x_))
prob_prob_inter = lam * prob1 + (1. - lam) * prob2

# interpolation on Gaussian
c_epsilon1, _ = model.prior.cNF(model.ae.c_encode(x))
c_epsilon2, _ = model.prior.cNF(model.ae.c_encode(x_))
c_interpolation = tf.nn.softmax(model.prior.cflow(lam * c_epsilon1 + (1. - lam) * c_epsilon2))
#%%
non_smooth_gap = np.zeros((10, ))
prob_gap = np.zeros((10, ))
flow_gap = np.zeros((10, ))
for idx in range(5000):
    non_smooth_gap += np.abs((prob_x_inter.numpy()[idx] - prob_c_inter.numpy()[idx])) / 5000
    prob_gap += np.abs((prob_x_inter.numpy()[idx] - prob_prob_inter.numpy()[idx])) / 5000
    flow_gap += np.abs((prob_x_inter.numpy()[idx] - c_interpolation.numpy()[idx])) / 5000

plt.figure(figsize=(7, 4))
plt.plot(np.arange(10), non_smooth_gap, label='non-smooth')
plt.plot(np.arange(10), prob_gap, label='softmax')
plt.plot(np.arange(10), flow_gap, label='flow')
plt.legend()
plt.show()
#%%
'''interpolation: z'''
lam = 0.2
tf.random.set_seed(1)
x_ = tf.gather(x, tf.random.shuffle(tf.range(tf.shape(x)[0])))
x_inter = lam * x + (1. - lam) * x_

# interpolation on x
z_x_inter = model.ae.z_encode(x_inter)

# interpolation on z
z1 = model.ae.z_encode(x)
z2 = model.ae.z_encode(x_)
z_z_inter = lam * z1 + (1. - lam) * z2

# interpolation on Gaussian
z_epsilon1, _ = model.prior.zNF(z1)
z_epsilon2, _ = model.prior.zNF(z2)
z_interpolation = model.prior.zflow(lam * z_epsilon1 + (1. - lam) * z_epsilon2)
#%%
non_smooth_gap = np.zeros((6, ))
flow_gap = np.zeros((6, ))
for idx in range(5000):
    non_smooth_gap += np.abs((z_x_inter.numpy()[idx] - z_z_inter.numpy()[idx]) / z_x_inter.numpy()[idx]) / 5000
    flow_gap += np.abs((z_x_inter.numpy()[idx] - z_interpolation.numpy()[idx]) / z_x_inter.numpy()[idx]) / 5000

plt.figure(figsize=(7, 3))
plt.plot(np.arange(6), non_smooth_gap, label='non-smooth')
plt.plot(np.arange(6), flow_gap, label='flow')
plt.legend()
plt.show()
#%%