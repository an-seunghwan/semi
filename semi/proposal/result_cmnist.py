#%%
import argparse
import os

os.chdir(r'D:\semi\semi\proposal') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/semi/proposal') # main directory (repository)

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
from model2_cmnist import VAE
# from criterion import ELBO_criterion
from mixup import augment
#%%
import ast
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
#%%
def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='cmnist')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results (ex. generating color MNIST)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=300, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=50, type=int,
                        metavar='N', help='reconstruct frequency (default: 50)')
    parser.add_argument('--labeled_examples', type=int, default=100, 
                        help='number labeled examples (default: 100')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    parser.add_argument('--augment', action='store_true', 
                        help="apply augmentation to image")

    '''Deep VAE Model Parameters'''
    # parser.add_argument('--net-name', default="wideresnet-28-2", type=str, help="the name for network to use")
    # parser.add_argument('--depth', type=int, default=28, 
    #                     help='depth for WideResnet (default: 28)')
    # parser.add_argument('--width', type=int, default=2, 
    #                     help='widen factor for WideResnet (default: 2)')
    # parser.add_argument('--slope', type=float, default=0.1, 
    #                     help='slope parameter for LeakyReLU (default: 0.1)')
    # parser.add_argument('-dr', '--drop_rate', default=0, type=float, 
    #                     help='drop rate for the network')
    parser.add_argument("--br", "--bce_reconstruction", action='store_true', 
                        help='Do BCE Reconstruction')
    parser.add_argument("-s", "--x_sigma", default=0.5, type=float,
                        help="The standard variance for reconstructed images, work as regularization")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=6, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--mixup_max_z', default=10, type=float, 
                        help='the epoch to linear adjust mixup')
    parser.add_argument('--mixup_epoch_z',default=1, type=int, 
                        help='the max epoch to adjust mixup')
    parser.add_argument('--mixup_max_y', default=10, type=float, 
                        help='the epoch to linear adjust mixup')
    parser.add_argument('--mixup_epoch_y',default=1, type=int, 
                        help='the max epoch to adjust mixup')
    parser.add_argument('--lambda',default=10, type=float, 
                        help='the weight of classification and mutual information')
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    # parser.add_argument('-b1', '--beta1', default=0.9, type=float, metavar='Beta1 In ADAM and SGD',
    #                     help='beta1 for adam as well as momentum for SGD')
    parser.add_argument('-ad', "--adjust_lr", default=[100, 200], type=arg_as_list,
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float)

    '''Normalizing Flow Model Parameters'''
    parser.add_argument('--z_hidden_dim', default=64, type=int,
                        help='embedding dimension of continuous latent for coupling layer')
    parser.add_argument('--c_hidden_dim', default=64, type=int,
                        help='embedding dimension of discrete latent for coupling layer')
    parser.add_argument('--z_n_blocks', default=3, type=int,
                        help='number of coupling layers in Real NVP (continous latent)')
    parser.add_argument('--c_n_blocks', default=4, type=int,
                        help='number of coupling layers in Real NVP (discrete latent)')

    '''Normalizing Flow Optimizer Parameters'''
    parser.add_argument('--lr_nf', '--learning-rate-nf', default=0.001, type=float,
                        metavar='LR', help='initial learning rate for normalizing flow')
    parser.add_argument('--lr_gamma_nf', default=0.1, type=float)
    parser.add_argument('--wd_nf', '--weight-decay-nf', default=2e-5, type=float,
                        help='L2 regularization parameter for dense layers in Real NVP')
    parser.add_argument('-b1_nf', '--beta1_nf', default=0.9, type=float, metavar='Beta1 In ADAM',
                        help='beta1 for adam')
    parser.add_argument('-b2_nf', '--beta2_nf', default=0.99, type=float, metavar='Beta2 In ADAM',
                        help='beta2 for adam')
    parser.add_argument('-ad_nf', "--adjust_lr_nf", default=[0.25, 0.5, 0.75], type=arg_as_list,
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--start_epoch_nf', default=0, type=int,
                        help="NF training start epoch")
    # parser.add_argument('--decay_steps', default=1, type=int,
    #                     help='decay steps for exponential decay schedule')
    # parser.add_argument('--decay_rate', default=0.95, type=float,
    #                     help='decay rate for exponential decay schedule')
    # parser.add_argument('--gradclip', default=1., type=float,
    #                     help='gradclip value')

    '''Optimizer Transport Estimation Parameters'''
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="the label smoothing epsilon for labeled data")

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
args = vars(get_args().parse_args(args=['--config_path', 'configs/cmnist_100.yaml']))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)

model_path = log_path + '/20220205-143742'
model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
model = VAE(args, num_classes)
# model = AutoEncoder(num_classes=num_classes,
#                         latent_dim=args['latent_dim'], 
#                         output_channel=3, 
#                         activation='sigmoid',
#                         input_shape=(None, 32, 32, 3))
model.build(input_shape=(None, 32, 32, 3))
model.load_weights(model_path + '/' + model_name)
model.summary()
#%%
'''style transfer'''
x = []
y = []
for example in datasetU:
    x.append(example[0])
    y.append(example[1])
    if len(x) == 100: break
x = tf.cast(np.array(x), tf.float32)
y = tf.cast(np.array(y), tf.float32)
latent = model.ae.z_encode(x, training=False)
#%%
for idx in tqdm.tqdm(range(100)):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, num_classes+1, 1)
    plt.imshow(x[idx])
    plt.title('original')
    plt.axis('off')
    
    for i in range(num_classes):
        label = np.zeros((1, num_classes))
        label[:, i] = 1
        xhat = model.ae.decode(latent.numpy()[[idx]], label, training=False)

        plt.subplot(1, num_classes+1, i+2)
        plt.imshow(xhat[0])
        plt.title('given label {}'.format(i))
        plt.axis('off')
    plt.savefig('{}/img{}.png'.format(model_path, idx),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    # plt.show()
    plt.close()
#%%
style = []
for idx in [1, 2, 7, 12, 22, 27, 41, 45, 74, 87]:
    style.append(Image.open('{}/img{}.png'.format(model_path, idx)))
    
fig, axes = plt.subplots(10, 1, figsize=(10, 6))
for i in range(len(style)):
    axes.flatten()[i].imshow(style[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/style_transfer.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''interpolation: smooth'''
z_epsilon1, _ = model.prior.zNF(latent.numpy()[[10], :])
z_epsilon2, _ = model.prior.zNF(latent.numpy()[[61], :])

interpolation = np.squeeze(np.linspace(z_epsilon1, z_epsilon2, 15))
z_interpolation = model.prior.zflow(interpolation)
# out = model.prior.zNF.affine_layers[-1].reverse(interpolation)
# for i in range(model.prior.zNF.n_blocks - 1):
#     out = model.prior.zNF.permutations[-1-i].reverse(out)
#     out = model.prior.zNF.affine_layers[-2-i].reverse(out)
# z_interpolation = out

label = np.zeros((z_interpolation.shape[0], num_classes))
label[:, 6] = 1
xhat_ = model.ae.decode(z_interpolation, label, training=False)

fig, axes = plt.subplots(1, 15, figsize=(15, 6))
for i in range(len(z_interpolation)):
    axes.flatten()[i].imshow(xhat_[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/style_interpolation_smooth.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''interpolation: non-smooth'''
z_interpolation = np.squeeze(np.linspace(latent.numpy()[[10], :], 
                                         latent.numpy()[[61], :], 15))

label = np.zeros((z_interpolation.shape[0], num_classes))
label[:, 6] = 1
xhat_ = model.ae.decode(z_interpolation, label, training=False)

fig, axes = plt.subplots(1, 15, figsize=(15, 6))
for i in range(len(z_interpolation)):
    axes.flatten()[i].imshow(xhat_[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/style_interpolation_nonsmooth.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''manipulation'''
idx = [1, 2, 7, 12, 22, 27, 41, 45, 74, 87]
plt.figure(figsize=(15, 20))
for j in range(len(idx)):
    z = model.ae.z_encode(tf.cast(x.numpy()[[idx[j]]], tf.float32), training=False)
    p = np.linspace(1, 0, 11)
    num1 = 4
    num2 = 9

    for i in range(len(p)):
        attr = np.zeros((1, num_classes))
        attr[:, num1] = p[i]
        attr[:, num2] = 1 - p[i]
        xhat = model.ae.decode(z, attr, training=False)

        plt.subplot(len(idx), len(p), len(p) * j + i + 1)
        plt.imshow(xhat[0])
        plt.title('p of {}: {:.1f}'.format(num1, p[i]))
        plt.axis('off')
plt.savefig('{}/manipulation.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''style latent random sampling of c'''
tf.random.set_seed(1)
z = model.ae.z_encode(tf.cast(x.numpy()[[idx[1]]], tf.float32), training=False)
c_epsilon = tf.random.normal(shape=(100, num_classes))
c = model.prior.cflow(c_epsilon)
# out = model.prior.cNF.affine_layers[-1].reverse(c_epsilon)
# for i in range(model.prior.cNF.n_blocks - 1):
#     out = model.prior.cNF.permutations[-1-i].reverse(out)
#     out = model.prior.cNF.affine_layers[-2-i].reverse(out)
# c = out
pi = tf.nn.softmax(c)
xhat = model.ae.decode(tf.tile(z, (len(pi), 1)), pi, training=False)

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i in range(100):
    axes.flatten()[i].imshow(xhat[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/inverse_flow_c.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''style latent random sampling of z'''
tf.random.set_seed(1)
z_epsilon = tf.random.normal(shape=(100, args['latent_dim']))
z = model.prior.zflow(z_epsilon)
# out = model.prior.zNF.affine_layers[-1].reverse(z_epsilon)
# for i in range(model.prior.zNF.n_blocks - 1):
#     out = model.prior.zNF.permutations[-1-i].reverse(out)
#     out = model.prior.zNF.affine_layers[-2-i].reverse(out)
# z = out
pi = np.zeros((len(z), num_classes))
pi[:, 4] = 1
xhat = model.ae.decode(z, pi, training=False)

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i in range(100):
    axes.flatten()[i].imshow(xhat[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/inverse_flow_z.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
# np.mean(latent.numpy(), axis=0) # why sparse?
# np.mean(style_latent.numpy(), axis=0)
# np.mean(epsilon.numpy(), axis=0)
# np.mean(style_epsilon.numpy(), axis=0)
#%%