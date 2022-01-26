#%%
'''
custom SGD + weight decay ver.2

211222: lr schedule -> modify lr manually, instead of tensorflow function
211227: tf.abs -> tf.math.abs
211229: convert dmi -> tf.cast(dmi, tf.float32)
220101: convert dmi -> tf.constant(dmi, dtype=tf.float32)
220104: convert dmi -> tf.convert_to_tensor(dmi, dtype=tf.float32)
220104: monitoring KL-divergence and its absolute value
220107: decoupled weight decay https://arxiv.org/pdf/1711.05101.pdf
220110: modify mixup shuffle & optimal matching argument
220113: modify weight decay factor = weight decay * scheduled lr
'''
#%%
import argparse
import os

os.chdir(r'D:\semi\shotvae') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/shotvae') # main directory (repository)

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
from criterion2 import ELBO_criterion
from mixup import augment, optimal_match_mix, weight_decay_decoupled, label_smoothing 
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

    # parser.add_argument('-bp', '--base_path', default=".")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra)')
    # parser.add_argument('-is', "--image-size", default=[32, 32], type=arg_as_list,
    #                     metavar='Image Size List', help='the size of h * w for image')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SSL VAE Train PreProcess Parameter'''
    # parser.add_argument('-t', '--train-time', default=1, type=int,
    #                     metavar='N', help='the x-th time of training')
    parser.add_argument('--epochs', default=600, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    # parser.add_argument('--print-freq', '-p', default=3, type=int,
    #                     metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--reconstruct-freq', '-rf', default=50, type=int,
                        metavar='N', help='reconstruct frequency (default: 50)')
    # parser.add_argument('--annotated-ratio', default=0.1, type=float, help='The ratio for semi-supervised annotation')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters'''
    # parser.add_argument('--net-name', default="wideresnet-28-2", type=str, help="the name for network to use")
    parser.add_argument('--depth', type=int, default=28, 
                        help='depth for WideResnet (default: 28)')
    parser.add_argument('--width', type=int, default=2, 
                        help='widen factor for WideResnet (default: 2)')
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='slope parameter for LeakyReLU (default: 0.1)')
    parser.add_argument('--temperature', default=0.67, type=float,
                        help='centeralization parameter')
    parser.add_argument('-dr', '--drop-rate', default=0, type=float, 
                        help='drop rate for the network')
    parser.add_argument("--br", "--bce-reconstruction", action='store_true', 
                        help='Do BCE Reconstruction')
    parser.add_argument("-s", "--x-sigma", default=1, type=float,
                        help="The standard variance for reconstructed images, work as regularization")

    '''VAE parameters, notice we do not manually set the mutual information'''
    parser.add_argument('--ldc', "--latent-dim-continuous", default=128, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--cmi', "--continuous-mutual-info", default=0, type=float,
                        help='The mutual information bounding between x and the continuous variable z')
    parser.add_argument('--dmi', "--discrete-mutual-info", default=0, type=float,
                        help='The mutual information bounding between x and the discrete variable z')

    '''VAE Loss Function Parameters'''
    # parser.add_argument("-ei", "--evaluate-inference", action='store_true',
    #                     help='Calculate the inference accuracy for unlabeled dataset')
    parser.add_argument('--kbmc', '--kl-beta-max-continuous', default=1e-3, type=float, 
                        metavar='KL Beta', help='the epoch to linear adjust kl beta')
    parser.add_argument('--kbmd', '--kl-beta-max-discrete', default=1e-3, type=float, 
                        metavar='KL Beta', help='the epoch to linear adjust kl beta')
    parser.add_argument('--akb', '--adjust-kl-beta-epoch', default=200, type=int, 
                        metavar='KL Beta', help='the max epoch to adjust kl beta')
    parser.add_argument('--ewm', '--elbo-weight-max', default=1e-3, type=float, 
                        metavar='weight for elbo loss part')
    parser.add_argument('--aew', '--adjust-elbo-weight', default=400, type=int,
                        metavar="the epoch to adjust elbo weight to max")
    parser.add_argument('--wrd', default=1, type=float,
                        help="the max weight for the optimal transport estimation of discrete variable c")
    parser.add_argument('--wmf', '--weight-modify-factor', default=0.4, type=float,
                        help="weight  will get wrz at amf * epochs")
    parser.add_argument('--pwm', '--posterior-weight-max', default=1, type=float,
                        help="the max value for posterior weight")
    parser.add_argument('--apw', '--adjust-posterior-weight', default=200, type=float,
                        help="adjust posterior weight")

    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-b1', '--beta1', default=0.9, type=float, metavar='Beta1 In ADAM and SGD',
                        help='beta1 for adam as well as momentum for SGD')
    parser.add_argument('-ad', "--adjust-lr", default=[400, 500, 550], type=arg_as_list,
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)

    '''Optimizer Transport Estimation Parameters'''
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="the label smoothing epsilon for labeled data")
    parser.add_argument('--om', action='store_true', help="the optimal match for unlabeled data mixup")

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
args = vars(get_args().parse_args(args=['--config_path', 'configs/cifar10_4000.yaml']))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)

model_path = r'D:\semi\shotvae\logs\cifar10_4000\7.change_dataset\seed1\20220120-191853'
model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
model = VAE(num_classes=num_classes, depth=args['depth'], width=args['width'], slope=args['slope'],
            latent_dim=args['ldc'], temperature=args['temperature'])
model.build(input_shape=[(None, 32, 32, 3), (None, num_classes)])
model.load_weights(model_path + '/' + model_name)
model.summary()
#%%
shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)
iteratorL = iter(shuffle_and_batch(datasetL))
image, label = next(iteratorL)
#%%
data_dir = r'D:\cifar10_{}'.format(5000)
idx = np.arange(100)
x = np.array([np.load(data_dir + '/x_{}.npy'.format(i)) for i in idx])
y = np.array([np.load(data_dir + '/y_{}.npy'.format(i)) for i in idx])
x = tf.cast(x, tf.float32) / 255.
#%%
mean, log_sigma, log_prob, z, y, xhat = model([x, y])
#%%
'''interpolation'''
class_idx = 1
i = 0
j = 5
# class_idx = 7
# i = 0
# j = 2
interpolation_idx = np.where([tf.argmax(y, axis=-1).numpy() == class_idx])[1]
inter = np.linspace(z[interpolation_idx[i]], z[interpolation_idx[j]], 8)
inter_recon = model.decode_sample(inter, tf.one_hot([class_idx] * 8, depth=num_classes), training=False)

figure = plt.figure(figsize=(25, 5))
plt.subplot(1, 8+2, 1)
plt.imshow(x[interpolation_idx[i]])
plt.axis('off')
for i in range(8):
    plt.subplot(1, 8+2, i+2)
    plt.imshow(inter_recon[i])
    plt.axis('off')
plt.subplot(1, 8+2, 8+2)
plt.imshow(x[interpolation_idx[j]])
plt.axis('off')
plt.show()
#%%