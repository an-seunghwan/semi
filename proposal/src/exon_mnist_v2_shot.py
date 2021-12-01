#%%
import tensorflow as tf
import tensorflow.keras as K
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
# os.chdir(r'D:\EXoN')
# os.chdir('/Users/anseunghwan/Documents/GitHub/EXoN')
# os.chdir('/home/jeon/Desktop/an/EXoN')
os.chdir('/home1/prof/jeon/an/EXoN')

from modules import MNIST
#%%
PARAMS = {
    "data": 'mnist',
    "batch_size": 2000,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 2,
    "sigma": 4.,
    "activation": 'tanh',
    "observation": 'mse', # abs or mse
    "epochs": 1000, 
    "lambda1": 5000., 
    "lambda2": 2.,
    # "tau": 1.,
    "learning_rate": 0.001,
    "labeled": 1000,
    "hard": True,
    "FashionMNIST": False,
    "beta_trainable": True,
    "conceptual": 'circle', # circle or star
    "information": False,
}
#%%
'''triaining log file'''
from datetime import datetime
now = datetime.now() 
date_time = now.strftime("%y%m%d_%H%M")

import sys
sys.stdout = open("./assets/{}/log_{}_{}.txt".format(PARAMS['data'], PARAMS['data'], date_time), "w")
#%%
print(
'''
EXoN: EXplainable encoder Network v2
with MNIST dataset

1. negative cross-entropy -> KL-divergence with smoothed one-hot label
'''
)

from pprint import pprint
pprint(PARAMS)
#%%
if PARAMS['conceptual'] == 'circle':
    r = 2*np.sqrt(PARAMS['sigma']) / np.sin(np.pi / 10)
    prior_means = np.array([[r*np.cos(np.pi/10), r*np.sin(np.pi/10)],
                            [r*np.cos(3*np.pi/10), r*np.sin(3*np.pi/10)],
                            [r*np.cos(5*np.pi/10), r*np.sin(5*np.pi/10)],
                            [r*np.cos(7*np.pi/10), r*np.sin(7*np.pi/10)],
                            [r*np.cos(9*np.pi/10), r*np.sin(9*np.pi/10)],
                            [r*np.cos(11*np.pi/10), r*np.sin(11*np.pi/10)],
                            [r*np.cos(13*np.pi/10), r*np.sin(13*np.pi/10)],
                            [r*np.cos(15*np.pi/10), r*np.sin(15*np.pi/10)],
                            [r*np.cos(17*np.pi/10), r*np.sin(17*np.pi/10)],
                            [r*np.cos(19*np.pi/10), r*np.sin(19*np.pi/10)]])
elif PARAMS['conceptual'] == 'star':
    r = 4*np.sqrt(PARAMS['sigma'])
    prior_means = np.array([[0, -r*np.cos(np.pi/3)],
                            [2*r*np.cos(np.pi/6), r*np.sin(np.pi/3)],
                            [0, 2*r*np.cos(np.pi/6)/np.sin(np.pi/3)],
                            [-2*r*np.cos(np.pi/6), r*np.sin(np.pi/3)],
                            [-np.sqrt(2)*r*np.cos(np.pi/6), -2*r*np.cos(np.pi/3)],
                            [np.sqrt(2)*r*np.cos(np.pi/6), -2*r*np.cos(np.pi/3)],
                            [r*np.cos(np.pi/6), 0],
                            [r*np.cos(np.pi/6)*np.cos(np.pi/3), r*np.sin(np.pi/3)],
                            [-r*np.cos(np.pi/6)*np.cos(np.pi/3), r*np.sin(np.pi/3)],
                            [-r*np.cos(np.pi/6), 0]])
    
'''Figure 2 (Section 4.1)'''
np.random.seed(1)
samples = []
color = []
for i in range(len(prior_means)):
    samples.extend(np.random.multivariate_normal(mean=prior_means[i, :2], cov=np.array([[PARAMS['sigma'], 0], 
                                                                                        [0, PARAMS['sigma']]]), size=1000))
    color.extend([i] * 1000)
samples = np.array(samples)

plt.figure(figsize=(8, 8))
plt.tick_params(labelsize=25)   
plt.scatter(samples[:, 0], samples[:, 1], s=9, c=color, cmap=plt.cm.Reds, alpha=1)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
for i in range(PARAMS['class_num']):
    plt.text(prior_means[i, 0]-1, prior_means[i, 1]-1, "{}".format(i), fontsize=35)
    if i in [6, 7, 8, 9]:
        plt.text(prior_means[i, 0]-1, prior_means[i, 1]-1, "{}".format(i), fontsize=35, color='white')
plt.savefig('./assets/{}/prior_samples.png'.format(PARAMS['data']),
            bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()

# prior_means = np.tile(prior_means[np.newaxis, :, :], (PARAMS['batch_size'], 1, 1))
prior_means = tf.cast(prior_means[np.newaxis, :, :], tf.float32)
PARAMS['prior_means'] = prior_means
#%%
# data
if PARAMS['FashionMNIST']:
    (x_train, y_train), (x_test, y_test) = K.datasets.fashion_mnist.load_data()
else:
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])

np.random.seed(520)
# ensure that all classes are balanced 
lidx = np.concatenate([np.random.choice(np.where(y_train == i)[0], int(PARAMS['labeled'] / PARAMS['class_num']), replace=False) 
                        for i in range(PARAMS['class_num'])])
x_train_L = x_train[lidx]
y_train_L = y_train_onehot[lidx]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
train_dataset_L = tf.data.Dataset.from_tensor_slices((x_train_L, y_train_L)).shuffle(len(x_train_L), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(PARAMS['batch_size'])
#%%
model = MNIST.MixtureVAE(PARAMS)
optimizer = K.optimizers.Adam(PARAMS["learning_rate"])
#%%
@tf.function
def loss_mixture(prob, xhat, x, mean, logvar, beta, PARAMS):
    # reconstruction error
    if PARAMS['observation'] == 'abs':
        error = tf.reduce_mean(tf.reduce_sum(tf.math.abs(x - xhat), axis=-1))
    elif PARAMS['observation'] == 'mse':
        error = tf.reduce_mean(tf.reduce_sum(tf.math.square(x - xhat), axis=-1) / (2 * beta))
    
    # KL divergence by closed form
    kl = tf.reduce_mean(tf.reduce_sum(prob * tf.math.log(prob * PARAMS['class_num'] + 1e-8), axis=1))
    kl += tf.reduce_mean(tf.reduce_sum(tf.multiply(prob,
                                                  tf.reduce_sum(0.5 * (tf.math.pow(mean - PARAMS['prior_means'], 2) / PARAMS['sigma'] 
                                                                - 1
                                                                - tf.math.log(1 / PARAMS['sigma']) 
                                                                + tf.math.exp(logvar) / PARAMS['sigma']
                                                                - logvar), axis=-1)), axis=-1))
    
    return error, kl
#%%
@tf.function
def train_step(x_batch_L, y_batch_L, x_batch, beta, lambda1, lambda2):
    with tf.GradientTape() as tape:
        mean, logvar, prob, _, _, _, xhat = model(x_batch)
        error, kl = loss_mixture(prob, xhat, x_batch, mean, logvar, beta, PARAMS) 
        loss = error + kl
        
        prob_L = model.Classifier(x_batch_L)
        smooth = y_batch_L
        smooth *= 1. - 0.001
        smooth += 0.001 / PARAMS['class_num']
        cce = tf.reduce_mean(tf.reduce_sum(prob_L * (tf.math.log(prob_L + 1e-8) - tf.math.log(smooth + 1e-8)), axis=1))
        # cce = - tf.reduce_mean(tf.multiply(y_batch_L, tf.math.log(prob_L + 1e-20)))
        
        loss += (1. + tf.cast(lambda1, tf.float32)) * cce + (PARAMS['data_dim'] / 2) * tf.math.log(2 * np.pi * beta)                        
        # loss += (1. + tf.cast(lambda1, tf.float32)) * cce + (PARAMS['data_dim'] / 2) * tf.math.log(2 * np.pi * beta)
        
        if PARAMS['beta_trainable']:
            loss += tf.cast(lambda2, tf.float32) * (PARAMS['data_dim'] / 2) * (1 / beta)
                
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    
    return loss, error, kl, cce, xhat

@tf.function 
def beta_step(x_batch, beta):
    Dz = model(x_batch)[-1]
    beta += tf.reduce_sum(tf.reduce_mean(tf.math.square(x_batch - Dz), axis=-1))
    return beta
#%%
'''training'''
step = 0
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))

loss_ = []
error_ = []
kl_ = []
cce_ = []

train_error = []
test_error = []

'''train classification error'''
prob = model.Classifier(x_train)
error_count = len(np.where(np.squeeze(y_train) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
train_error.append(error_count / len(y_train))

'''test classification error''' 
error_count = 0
for x_batch, y_batch in test_dataset:
    prob = model.Classifier(x_batch)
    error_count += len(np.where(np.squeeze(y_batch) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
test_error.append(error_count / len(y_test))

if PARAMS['beta_trainable']:
    beta = tf.Variable(PARAMS["lambda2"], trainable=True, name="beta", dtype=tf.float32)
    betapath = []
else:
    beta = tf.Variable(PARAMS["lambda2"], trainable=False, name="beta", dtype=tf.float32)

for _ in progress_bar:
    x_batch = next(iter(train_dataset))
    x_batch_L, y_batch_L = next(iter(train_dataset_L))
    
    loss, error, kl, cce, xhat = train_step(x_batch_L, y_batch_L, x_batch, beta, PARAMS['lambda1'], PARAMS['lambda2'])
    
    loss_.append(loss.numpy())
    error_.append(error.numpy())
    kl_.append(kl.numpy())
    cce_.append(cce.numpy())

    if PARAMS['beta_trainable']:
        # beta training by Alternatning algorithm
        beta = 0
        for x_batch in train_dataset:
            beta = beta_step(x_batch, beta)
        beta = beta / len(x_train) + PARAMS['lambda2']
        betapath.append(beta.numpy())
    
    progress_bar.set_description('setting: batch_size={}, lr={:.5f}, labeled={}, lambda1={}, lambda2={}, sigma={}, beta={} | iteration {}/{} | loss {:.3f}, recon {:.3f}, kl {:.3f}, cce {:.3f}, train {:.3f}, test {:.3f}'.format(
        PARAMS['batch_size'], PARAMS['learning_rate'], PARAMS['labeled'], PARAMS['lambda1'], PARAMS['lambda2'], PARAMS['sigma'], beta,
        step, PARAMS['epochs'], 
        loss_[-1], error_[-1], kl_[-1], cce_[-1], train_error[-1], test_error[-1])) 
    
    step += 1
    
    if step % 100 == 0:
        '''train classification error'''
        prob = model.Classifier(x_train)
        error_count = len(np.where(np.squeeze(y_train) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
        train_error.append(error_count / len(y_train))
        
        '''test classification error''' 
        error_count = 0
        for x_batch, y_batch in test_dataset:
            prob = model.Classifier(x_batch)
            error_count += len(np.where(np.squeeze(y_batch) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
        test_error.append(error_count / len(y_test))
        
        print('setting: batch_size={}, lr={:.5f}, labeled={}, lambda1={}, lambda2={}, sigma={}, beta={} | iteration {}/{} | loss {:.3f}, recon {:.3f}, kl {:.3f}, cce {:.3f}, train {:.3f}, test {:.3f}'.format(
            PARAMS['batch_size'], PARAMS['learning_rate'], PARAMS['labeled'], PARAMS['lambda1'], PARAMS['lambda2'], PARAMS['sigma'], beta,
            step, PARAMS['epochs'], 
            loss_[-1], error_[-1], kl_[-1], cce_[-1], train_error[-1], test_error[-1]))
        
    if step == PARAMS['epochs']: break
#%%
asset_path = 'weights_{}_{}_v2'.format(PARAMS['data'], date_time)
model.save_weights('./assets/{}/{}/weights'.format(PARAMS['data'], asset_path))
#%%
# model = MNIST.MixtureVAE(PARAMS)
# asset_path = 'weights_{}_{}_v2'.format(PARAMS['data'], '210824_1515')
# model.load_weights('./assets/{}/{}/weights'.format(PARAMS['data'], asset_path))
#%%
mean_, logvar_, prob, _, _, _, xhat_ = model(x_test)
'''classification loss'''
classification_error = len(np.where(y_test - np.argmax(prob.numpy(), axis=-1) != 0)[0]) / len(y_test)
print('classification loss: ', classification_error)
#%%
'''KL divergence'''
_, kl = loss_mixture(prob, xhat_, x_test, mean_, logvar_, beta, PARAMS) 
print('KL divergence: ', kl.numpy())
#%%
# '''negative SSIM'''
# a = np.arange(-15, 15.1, 1.0)
# b = np.arange(-15, 15.1, 1.0)
# aa, bb = np.meshgrid(a, b, sparse=True)
# grid = []
# for b_ in reversed(bb[:, 0]):
#     for a_ in aa[0, :]:
#         grid.append(np.array([a_, b_]))
# grid_output = model.decoder(tf.cast(np.array(grid), tf.float32))
# ssim = 0
# for i in tqdm(range(len(grid_output))):
#     s = tf.image.ssim(tf.reshape(grid_output[i, :], (28, 28, 1)), tf.reshape(grid_output, (len(grid_output), 28, 28, 1)), 
#                     max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
#     ssim += np.sum(s.numpy())
# neg_ssim = (1 - ssim / (len(grid_output)*len(grid_output))) / 2
# print('negative SSIM: ', neg_ssim)
#%%
a = np.arange(-15, 15.1, 2.5)
b = np.arange(-15, 15.1, 2.5)
aa, bb = np.meshgrid(a, b, sparse=True)
grid = []
for b_ in reversed(bb[:, 0]):
    for a_ in aa[0, :]:
        grid.append(np.array([a_, b_]))
plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=26)    
plt.scatter(np.array(grid)[:, 0], np.array(grid)[:, 1], s=50, color='black')
plt.savefig('./assets/{}/{}/grid.png'.format(PARAMS['data'], asset_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''Figure 4 middle panel (Section 4.1)'''
grid_output = model.decoder(tf.cast(np.array(grid), tf.float32))
grid_output = grid_output.numpy()
plt.figure(figsize=(10, 10))
for i in range(len(grid)):
    plt.subplot(len(b), len(a), i+1)
    plt.imshow(grid_output[i].reshape(28, 28), cmap='gray')    
    plt.axis('off')
    plt.tight_layout() 
plt.savefig('./assets/{}/{}/reconstruction.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
reconstruction = Image.open('./assets/{}/{}/reconstruction.png'.format(PARAMS['data'], asset_path))

plt.figure(figsize=(10, 10))
plt.xticks(np.arange(-15, 15.1, 5))    
plt.yticks(np.arange(-15, 15.1, 5))    
plt.tick_params(labelsize=30)    
plt.imshow(reconstruction, extent=[-16.3, 16.3, -16.3, 16.3])
plt.tight_layout() 
plt.savefig('./assets/{}/{}/reconstruction.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''Figure 4 left panel (Section 4.1)'''
zmat = []
mean, logvar, logits, y, z, z_tilde, xhat = model(x_test)
zmat.extend(z_tilde.numpy().reshape(-1, PARAMS['latent_dim']))
zmat = np.array(zmat)
plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(zmat[:, 0], zmat[:, 1], c=y_test, s=10, cmap=plt.cm.Reds, alpha=1)
plt.savefig('./assets/{}/{}/latent.png'.format(PARAMS['data'], asset_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''Figure 4 right panel (Section 4.1)'''
a = np.arange(-20, 20.1, 0.25)
b = np.arange(-20, 20.1, 0.25)
aa, bb = np.meshgrid(a, b, sparse=True)
grid = []
for b_ in reversed(bb[:, 0]):
    for a_ in aa[0, :]:
        grid.append(np.array([a_, b_]))
grid = tf.cast(np.array(grid), tf.float32)
grid_output = model.decoder(grid)
grid_output = grid_output.numpy()
grid_prob = model.Classifier(grid_output)
grid_prob_argmax = np.argmax(grid_prob.numpy(), axis=1)
plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(grid[:, 0], grid[:, 1], c=grid_prob_argmax, s=10, cmap=plt.cm.Reds, alpha=1)
plt.savefig('./assets/{}/{}/conditional_prob.png'.format(PARAMS['data'], asset_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''loss phase'''
plt.rc('xtick', labelsize=10)   
plt.rc('ytick', labelsize=10)   
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(loss_, label='loss')
leg = ax.legend(fontsize=15, loc='lower right')
plt.savefig('./assets/{}/{}/loss_phase.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''error phase'''
plt.rc('xtick', labelsize=10)   
plt.rc('ytick', labelsize=10)   
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(error_, label='error')
leg = ax.legend(fontsize=15, loc='lower right')
plt.savefig('./assets/{}/{}/error_phase.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''KL phase'''
plt.rc('xtick', labelsize=10)   
plt.rc('ytick', labelsize=10)   
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(kl_, label='KL')
leg = ax.legend(fontsize=15, loc='lower right')
plt.savefig('./assets/{}/{}/kl_phase.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''CCE phase'''
plt.rc('xtick', labelsize=10)   
plt.rc('ytick', labelsize=10)   
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(cce_, label='CCE')
leg = ax.legend(fontsize=15, loc='lower right')
plt.savefig('./assets/{}/{}/cce_phase.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''beta phase'''
plt.rc('xtick', labelsize=10)   
plt.rc('ytick', labelsize=10)   
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(betapath, color='red', label='beta')
leg = ax.legend(fontsize=15, loc='lower right')
plt.savefig('./assets/{}/{}/beta_phase.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%