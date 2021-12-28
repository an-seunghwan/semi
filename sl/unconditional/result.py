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

model_path = log_path + '/20211228-131009'
model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
model = VAE(args, num_classes)
model.build(input_shape=(None, 32, 32, 3))
model.load_weights(model_path + '/' + model_name)
model.summary()
#%%
'''style transfer'''
x = []
y = []
for example in dataset:
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
for idx in [5, 16, 17, 30, 46, 58, 69, 80, 92, 97]:
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
'''interpolation'''
z_epsilon1, _ = model.prior.zNF(latent.numpy()[[17], :])
z_epsilon2, _ = model.prior.zNF(latent.numpy()[[30], :])

interpolation = np.squeeze(np.linspace(z_epsilon1, z_epsilon2, 20))
z_interpolation = model.prior.zflow(interpolation)

label = np.zeros((z_interpolation.shape[0], num_classes))
label[:, 3] = 1
xhat_ = model.ae.decode(z_interpolation, label, training=False)

fig, axes = plt.subplots(1, 20, figsize=(20, 6))
for i in range(len(z_interpolation)):
    axes.flatten()[i].imshow(xhat_[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/style_interpolation.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''manipulation'''
idx = [15, 84, 26, 45, 46, 52, 97, 54, 58, 92]
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
# plt.show()
plt.close()
#%%
'''style latent random sampling of c'''
tf.random.set_seed(1)
z = model.ae.z_encode(tf.cast(x.numpy()[[idx[0]]], tf.float32), training=False)
c_epsilon = tf.random.normal(shape=(100, num_classes))
c = model.prior.cflow(c_epsilon)
pi = tf.nn.softmax(c)
xhat = model.ae.decode(tf.tile(z, (len(pi), 1)), pi, training=False)

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i in range(100):
    axes.flatten()[i].imshow(xhat[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/inverse_flow_c.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''style latent random sampling of z'''
tf.random.set_seed(1)
z_epsilon = tf.random.normal(shape=(100, args['latent_dim']))
z = model.prior.zflow(z_epsilon)
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
# plt.show()
plt.close()
#%%
# np.mean(latent.numpy(), axis=0) # why sparse?
# np.mean(style_latent.numpy(), axis=0)
# np.mean(epsilon.numpy(), axis=0)
# np.mean(style_epsilon.numpy(), axis=0)
#%%