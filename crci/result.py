#%%
import argparse
import os

os.chdir(r'D:\semi\crci') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/crci') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm
import yaml
import io
import matplotlib.pyplot as plt
from PIL import Image

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import VAE
from criterion import ELBO_criterion
from utils import CustomReduceLRoP
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

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset used for training')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--labeled-batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size for labeled dataset (default: 32)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=200, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000), all labels are balanced')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters'''
    parser.add_argument("--bce_reconstruction", default=True, type=bool,
                        help="Do BCE Reconstruction")

    '''VAE parameters'''
    parser.add_argument('--z_dim', default=64, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--u_dim', default=64, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--depth', type=int, default=28, 
                        help='depth for WideResnet (default: 28)')
    parser.add_argument('--width', type=int, default=2, 
                        help='widen factor for WideResnet (default: 2)')
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='slope parameter for LeakyReLU (default: 0.1)')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--classifier_learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate for classifier')
    # parser.add_argument('--weight_decay', default=5e-4, type=float)
    
    parser.add_argument("--z_capacity", default=[0., 7., 100000, 15.], type=arg_as_list,
                        help="controlled capacity")
    parser.add_argument("--u_capacity", default=[0., 7., 100000, 15.], type=arg_as_list,
                        help="controlled capacity")
    parser.add_argument('--gamma_c', default=15, type=float,
                        help='weight of loss')
    parser.add_argument('--gamma_h', default=30, type=float,
                        help='weight of loss')
    parser.add_argument('--gamma_bc', default=30, type=float,
                        help='weight of loss')
    parser.add_argument('--bc_threshold', default=0.15, type=float,
                        help='threshold of Bhattacharyya coefficient')

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

model_path = log_path + '/20220411-135844'
model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]

model = VAE(
    num_classes=num_classes,
    latent_dim=args['z_dim'], 
    u_dim=args['u_dim'],
    depth=args['depth'], width=args['width'], slope=args['slope']
)
model.build(input_shape=(None, 32, 32, 3))
model.load_weights(model_path + '/' + model_name)
model.summary()
#%%
autotune = tf.data.AUTOTUNE
batch = lambda dataset: dataset.batch(batch_size=args['batch_size'], drop_remainder=False).prefetch(autotune)
iterator_test = iter(batch(test_dataset))
total_length = sum(1 for _ in test_dataset)
iteration = total_length // args['batch_size'] 
#%%
error_count = 0
for x_test_batch, y_test_batch in batch(test_dataset):
    prob = model.classify(x_test_batch, training=False)
    error_count += np.sum(tf.argmax(prob, axis=-1).numpy() - tf.argmax(y_test_batch, axis=-1).numpy() != 0)
print('TEST classification error: {:.2f}%'.format(error_count / total_length * 100))
#%%
z_means = []
z_logvars = []
u_means = []
u_logvars = []
labels = []
zs = []
us = []
probs = []
for i in tqdm.tqdm(range(iteration + 1)):
    image, label = next(iterator_test)
    z_mean, z_logvar, z, u_mean, u_logvar, u = model.encode(image, training=False)
    prob = model.classify(image, training=False)
    z_means.extend(z_mean)
    z_logvars.extend(z_logvar)
    u_means.extend(u_mean)
    u_logvars.extend(u_logvar)
    labels.extend(label)
    zs.extend(z)
    us.extend(u)
    probs.extend(prob)
z_means = tf.stack(z_means, axis=0)
z_logvars = tf.stack(z_logvars, axis=0)
u_means = tf.stack(u_means, axis=0)
u_logvars = tf.stack(u_logvars, axis=0)
labels = tf.stack(labels, axis=0)
zs = tf.stack(zs, axis=0)
us = tf.stack(us, axis=0)
probs = tf.stack(probs, axis=0)
#%%
'''KL-divergence'''
u_means_ = tf.tile(u_means[:, tf.newaxis, :], (1, num_classes, 1))
u_logvars_ = tf.tile(u_logvars[:, tf.newaxis, :], (1, num_classes, 1))
u_kl = tf.reduce_sum(0.5 * (tf.math.pow(u_means_ - model.u_prior_means, 2) / tf.math.exp(model.u_prior_logvars)
                            - 1
                            + tf.math.exp(u_logvars_) / tf.math.exp(model.u_prior_logvars)
                            + model.u_prior_logvars
                            - u_logvars_), axis=-1)
u_kl = tf.reduce_mean(tf.reduce_sum(tf.multiply(probs, u_kl), axis=-1))
print('KL-divergence of u latent: ', u_kl.numpy())
#%%
tf.random.set_seed(1)
z_rand = tf.random.normal((49, args['z_dim']))
for k in range(num_classes):
    grid_output = model.decoder(tf.concat([z_rand,
                                        tf.tile(model.u_prior_means.numpy()[[k], :], (z_rand.shape[0], 1))], axis=-1), training=False)
    grid_output = grid_output.numpy()
    plt.figure(figsize=(7, 7))
    for i in range(z_rand.shape[0]):
        plt.subplot(7, 7, i+1)
        plt.imshow(grid_output[i])    
        plt.axis('off')
        plt.tight_layout() 
    plt.savefig('./{}/recon_z_random_u{}_priormean.png'.format(model_path, k),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()
#%%
for k in range(num_classes):
    u_rand = np.random.multivariate_normal(mean=model.u_prior_means[k, :], 
                                        cov=np.diag(tf.math.exp(model.u_prior_logvars[k, :])),
                                        size=100)
    grid_output = model.decoder(tf.concat([tf.zeros((100, args['z_dim'])),
                                            u_rand], axis=-1), training=False)
    grid_output = grid_output.numpy()
    plt.figure(figsize=(10, 10))
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(grid_output[i])    
        plt.axis('off')
        plt.tight_layout() 
    plt.savefig('./{}/recon_z_priormean_u{}_random.png'.format(model_path, k),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()
#%%