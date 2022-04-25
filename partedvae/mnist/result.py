#%%
import argparse
import os

os.chdir(r'D:\semi\partedvae') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/partedvae') # main directory (repository)

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
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--labeled-batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size for labeled dataset (default: 32)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=80, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=100, 
                        help='number labeled examples (default: 100), all labels are balanced')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters'''
    parser.add_argument("--bce_reconstruction", default=True, type=bool,
                        help="Do BCE Reconstruction")

    '''VAE parameters'''
    parser.add_argument('--z_dim', default=6, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--u_dim', default=10, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--classifier_learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate for classifier')
    
    parser.add_argument("--z_capacity", default=[0., 7., 100000, 15.], type=arg_as_list,
                        help="controlled capacity") # cap_min, cap_max, num_iters, gamma
    parser.add_argument("--u_capacity", default=[0., 7., 100000, 15.], type=arg_as_list,
                        help="controlled capacity") # cap_min, cap_max, num_iters, gamma
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
args = vars(get_args().parse_args(args=['--config_path', 'configs/mnist_100.yaml']))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)

model_path = log_path + '{}/model'.format('20220425-220507')
model = K.models.load_model(model_path)
model.summary()
# model_path = log_path + '/20220425-200535'
# model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
# model = VAE(num_classes=num_classes,
#             latent_dim=args['z_dim'], 
#             u_dim=args['u_dim'])
# model.build(input_shape=(None, 28, 28, 1))
# model.load_weights(model_path + '/' + model_name)
# model.summary()
#%%
'''prior distributions'''
u_prior_means = model.u_prior_means
u_prior_logvars = 2. * tf.nn.tanh(model.u_prior_logvars_before_tanh) - 1.
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
'''Bhattacharyya coefficient'''
BC_valid_mask = np.ones((num_classes, num_classes))
BC_valid_mask = tf.constant(np.tril(BC_valid_mask, k=-1), tf.float32)

u_var = tf.math.exp(u_prior_logvars)
avg_u_var = 0.5 * (u_var[tf.newaxis, ...] + u_var[:, tf.newaxis, :])
inv_avg_u_var = 1. / (avg_u_var + 1e-8)
diff_mean = u_prior_means[tf.newaxis, ...] - u_prior_means[:, tf.newaxis, :]
D = 1/8 * tf.reduce_sum(diff_mean * inv_avg_u_var * diff_mean, axis=-1)
D += 0.5 * tf.reduce_sum(tf.math.log(avg_u_var + 1e-8), axis=-1)
D += - 0.25 * (tf.reduce_sum(u_prior_logvars, axis=-1)[tf.newaxis, ...] + tf.reduce_sum(u_prior_logvars, axis=-1)[:, tf.newaxis])
BC = tf.math.exp(- D)
valid_BC = BC * BC_valid_mask
valid_BC = tf.clip_by_value(valid_BC - args['bc_threshold'], 0., 1.)
BC_loss = tf.reduce_sum(valid_BC)
print('Bhattacharyya coefficient: {:.3f}%'.format(BC_loss))
#%%
with open("{}/result.txt".format(model_path), "w") as file:
    file.write(
        "TEST classification error: {:.2f}%\n\n".format(
            error_count / total_length * 100
        )
    )
    file.write('Bhattacharyya coefficient: {:.3f}'.format(BC_loss))
#%%
z_means = []
z_logvars = []
u_means = []
u_logvars = []
labels = []
zs = []
us = []
for i in tqdm.tqdm(range(iteration + 1)):
    image, label = next(iterator_test)
    z_mean, z_logvar, z, u_mean, u_logvar, u = model.encode(image, training=False)
    z_means.extend(z_mean)
    z_logvars.extend(z_logvar)
    u_means.extend(u_mean)
    u_logvars.extend(u_logvar)
    labels.extend(label)
    zs.extend(z)
    us.extend(u)
z_means = tf.stack(z_means, axis=0)
z_logvars = tf.stack(z_logvars, axis=0)
u_means = tf.stack(u_means, axis=0)
u_logvars = tf.stack(u_logvars, axis=0)
labels = tf.stack(labels, axis=0)
zs = tf.stack(zs, axis=0)
us = tf.stack(us, axis=0)
#%%
zmat = np.array(zs)
plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(zmat[:, 0], zmat[:, 1], c=tf.argmax(labels, axis=1).numpy(), s=10, cmap=plt.cm.Reds, alpha=1)
plt.savefig('./{}/test_dataset_conditional_sampled_latent_z.png'.format(model_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
umat = np.array(us)
plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(umat[:, 0], umat[:, 1], c=tf.argmax(labels, axis=1).numpy(), s=10, cmap=plt.cm.Reds, alpha=1)
plt.savefig('./{}/test_dataset_conditional_sampled_latent_u.png'.format(model_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''prior distribution (learned)'''
np.random.seed(1)
samples = []
color = []
for i in range(u_prior_means.shape[0]):
    samples.extend(np.random.multivariate_normal(mean=u_prior_means[i, :], 
                                                cov=np.array([[tf.math.exp(u_prior_logvars[i, 0]), 0], 
                                                            [0, tf.math.exp(u_prior_logvars[i, 1])]]), size=1000))
    color.extend([i] * 1000)
samples = np.array(samples)

plt.figure(figsize=(8, 8))
plt.tick_params(labelsize=25)   
plt.scatter(samples[:, 0], samples[:, 1], s=9, c=color, cmap=plt.cm.Reds, alpha=1)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
for i in range(num_classes):
    plt.text(u_prior_means[i, 0], u_prior_means[i, 1], "{}".format(i), fontsize=35)
    if i in [6, 7, 8, 9]:
        plt.text(u_prior_means[i, 0], u_prior_means[i, 1], "{}".format(i), fontsize=35, color='white')
plt.savefig('./{}/prior_samples.png'.format(model_path),
            bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
# '''test dataset mean and logvar of posterior distribution'''
# test_posterior_mean = []
# test_posterior_var = []
# for i in range(num_classes):
#     test_posterior_mean.append(umat[tf.argmax(labels, axis=1) == i, :].mean(axis=0))
#     test_posterior_var.append(umat[tf.argmax(labels, axis=1) == i, :].var(axis=0))
# test_posterior_mean = np.array(test_posterior_mean)
# test_posterior_var = np.array(test_posterior_var)
# #%%
# plt.plot(tf.math.abs(test_posterior_mean[:, 0] - u_prior_means[:, 0]), label="u dim 1 mean")
# plt.plot(tf.math.abs(test_posterior_mean[:, 1] - u_prior_means[:, 1]), label="u dim 2 mean")
# plt.legend()
# plt.savefig('./{}/u_prior_posterior_mean_gap.png'.format(model_path),
#                 dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
# plt.close()
# #%%
# plt.plot(tf.math.abs(test_posterior_var[:, 0] - tf.math.exp(u_prior_logvars)[:, 0]), label="u dim 1 var")
# plt.plot(tf.math.abs(test_posterior_var[:, 1] - tf.math.exp(u_prior_logvars)[:, 1]), label="u dim 2 var")
# plt.legend()
# plt.savefig('./{}/u_prior_posterior_var_gap.png'.format(model_path),
#                 dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
# plt.close()
#%%
a = np.arange(-1, 1.1, 0.5)
b = np.arange(-1, 1.1, 0.5)
aa, bb = np.meshgrid(a, b, sparse=True)
grid = []
for b_ in reversed(bb[:, 0]):
    for a_ in aa[0, :]:
        grid.append(np.array([a_, b_]))
#%%
for k in range(num_classes):
    grid_output = model.decoder(tf.concat([tf.cast(np.array(grid), tf.float32),
                                        tf.tile(u_prior_means.numpy()[[k], :], (len(grid), 1))], axis=-1), training=False)
    grid_output = grid_output.numpy()
    plt.figure(figsize=(4, 4))
    for i in range(len(grid)):
        plt.subplot(len(b), len(a), i+1)
        plt.imshow(grid_output[i].reshape(28, 28), cmap='gray_r')    
        plt.axis('off')
        plt.tight_layout() 
    plt.savefig('./{}/recon_z_grid_u{}_priormean.png'.format(model_path, k),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()
#%%
a = np.arange(-0, 4.1, 0.4)
b = np.arange(-2, 2.1, 0.4)
aa, bb = np.meshgrid(a, b, sparse=True)
grid = []
for b_ in reversed(bb[:, 0]):
    for a_ in aa[0, :]:
        grid.append(np.array([a_, b_]))

grid_output = model.decoder(tf.concat([tf.cast(tf.tile([[0, 0]], (len(grid), 1)), tf.float32),
                                        tf.cast(np.array(grid), tf.float32)], axis=-1), training=False)
grid_output = grid_output.numpy()
plt.figure(figsize=(10, 10))
for i in range(len(grid)):
    plt.subplot(len(b), len(a), i+1)
    plt.imshow(grid_output[i].reshape(28, 28), cmap='gray_r')    
    plt.axis('off')
    plt.tight_layout() 
plt.savefig('./{}/recon_z_priormean_u_grid.png'.format(model_path, k),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''interpolation on latent space'''
z_inter = (u_prior_means.numpy()[0], u_prior_means.numpy()[1])    
np.random.seed(1)
samples = []
color = []
for i in range(num_classes):
    samples.extend(np.random.multivariate_normal(mean=u_prior_means.numpy()[i, :], 
                                                cov=np.array([[u_prior_logvars.numpy()[i, 0], 0], 
                                                            [0, u_prior_logvars.numpy()[i, 0]]]), size=1000))
    color.extend([i] * 1000)
samples = np.array(samples)
plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(zmat[:, 0], zmat[:, 1], c=tf.argmax(labels, axis=1).numpy(), s=10, cmap=plt.cm.Reds, alpha=1)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.scatter(z_inter[0][0], z_inter[0][1], color='blue', s=100)
plt.annotate('A', (z_inter[0][0], z_inter[0][1]), fontsize=30)
plt.scatter(z_inter[1][0], z_inter[1][1], color='blue', s=100)
plt.annotate('B', (z_inter[1][0], z_inter[1][1]), fontsize=30)
plt.plot((z_inter[0][0], z_inter[1][0]), (z_inter[0][1], z_inter[1][1]), color='black', linewidth=2, linestyle='--')
plt.xlabel("$z_0$", fontsize=30)
plt.ylabel("$z_1$", fontsize=30)
plt.savefig('./{}/interpolation_path.png'.format(model_path), 
            dpi=100, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''interpolation'''
inter = np.linspace(z_inter[0], z_inter[1], 10)
inter_recon = model.decoder(tf.concat([np.zeros((10, args['z_dim'])), inter], axis=-1))
figure = plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10+1, i+1)
    plt.imshow(inter_recon[i].numpy().reshape(28, 28), cmap='gray_r')
    plt.axis('off')
plt.savefig('./{}/interpolation_path_recon.png'.format(model_path), 
            dpi=100, bbox_inches="tight", pad_inches=0.1)
plt.show()
#%%
'''path: interpolation path and reconstruction'''
img = [Image.open('./{}/interpolation_path.png'.format(model_path)),
        Image.open('./{}/interpolation_path_recon.png'.format(model_path))]
f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 0.25]})
a0.imshow(img[0])    
a0.axis('off')
a1.imshow(img[1])    
a1.axis('off')
plt.tight_layout() 
plt.savefig('./{}/interpolation_path_and_recon.png'.format(model_path),
            dpi=100, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%