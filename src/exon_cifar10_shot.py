#%%
import tensorflow as tf
import tensorflow.keras as K
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from icecream import ic
import os
# os.chdir(r'D:\EXoN')
os.chdir('/home1/prof/jeon/an/EXoN')

from modules import CIFAR10_shot, FIXMATCH
#%%
PARAMS = {
    "data": 'cifar10',
    "batch_size": 64,
    "class_num": 10,
    "data_dim": 32,
    "latent_dim": 10+246,
    "sigma1": 0.1, 
    "sigma2": 1., 
    "channel": 3, 
    "activation": 'tanh',
    "observation": 'mse', # abs or mse
    "epochs": 10000, 
    # "tau": 5.0,
    "learning_rate": 0.001,
    "hard": True, # Gumbel-Max trick
    "lambda1": 1.,
    "lambda2": 0.01,
    "beta_trainable": False,
    "labeled": 2500,
    "ratio": 7, 
    "pseudo_label": True, 
    "optimal_interpolation": False, 
    "dist": 1.,
    "ema": True,
    "depth": 28, 
    "slope": 0.1,
    "widen_factor": 2,
    "threshold": 0.95,
}
#%%
# '''triaining log file'''
from datetime import datetime
now = datetime.now() 
date_time = now.strftime("%y%m%d_%H%M")

import sys
sys.stdout = open("./assets/{}/log_{}_{}.txt".format(PARAMS['data'], PARAMS['data'], date_time), "w")
#%%
print(
'''
EXoN: EXplainable encoder Network with SHOT-VAE
with CIFAR-10 dataset

1. without Gumbel-Softmax approximation
    - forward: Gumbel-Max reparametrization
    - backward: stop gradient w.r.t. categorical distribution
2. one-hot prior means
3. all batch-normalization
4. exponential moving average on weights
'''
)
#%%
# data_dir = r'D:\cifar10_{}'.format(PARAMS['labeled'])
data_dir = '/home1/prof/jeon/an/cifar10_{}'.format(PARAMS['labeled'])

from pprint import pprint
pprint(PARAMS)
#%%
classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classdict = {i:x for i,x in enumerate(classnames)}
#%%
prior_means = np.zeros((PARAMS['class_num'], PARAMS['latent_dim']))
prior_means[:, :PARAMS['class_num']] = np.eye(PARAMS['class_num']) * PARAMS['dist']

prior_means = tf.cast(prior_means[np.newaxis, :, :], tf.float32)
PARAMS['prior_means'] = prior_means

sigma_vector = np.ones((1, PARAMS['latent_dim'])) 
sigma_vector[0, :PARAMS['class_num']] = PARAMS['sigma1']
sigma_vector[0, PARAMS['class_num']:] = PARAMS['sigma2']
PARAMS["sigma_vector"] = tf.cast(sigma_vector, tf.float32)
#%%
'''data'''
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])

np.random.seed(1)
# ensure that all classes are balanced 
lidx = np.concatenate([np.random.choice(np.where(y_train == i)[0], int(PARAMS['labeled'] / PARAMS['class_num']), replace=False) 
                        for i in range(PARAMS['class_num'])])
uidx = np.array([x for x in np.arange(len(x_train)) if x not in lidx])
x_train_U = x_train[uidx]
y_train_U = y_train_onehot[uidx]
x_train_L = x_train[lidx]
y_train_L = y_train_onehot[lidx]
#%%
encoder = CIFAR10_shot.Encoder(PARAMS) 
encoder.summary()
decoder = CIFAR10_shot.Decoder(PARAMS) 
decoder.summary()
classifier = CIFAR10_shot.Classifier(PARAMS)
classifier.summary()

opt_encoder = K.optimizers.Adam(PARAMS["learning_rate"], beta_1=0.5, beta_2=0.9)
opt_decoder = K.optimizers.Adam(PARAMS["learning_rate"], beta_1=0.5, beta_2=0.9)
opt_classifier = K.optimizers.Adam(PARAMS["learning_rate"] * 3)

if PARAMS['ema']:
    ema_encoder = tf.train.ExponentialMovingAverage(decay=0.999)
    ema_decoder = tf.train.ExponentialMovingAverage(decay=0.999)
    ema_classifier = tf.train.ExponentialMovingAverage(decay=0.999)
#%%
'''augmentation'''
# strong
strong_aug = FIXMATCH.RandAugment(3)

# weak
weak_aug = K.Sequential([
  K.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  K.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='reflect'),
])
#%%
@tf.function
def loss_labeled(smooth, prob, xhat, x, mean, logvar, beta, PARAMS):
    # reconstruction error
    if PARAMS['observation'] == 'abs':
        error = tf.reduce_sum(tf.reduce_sum(tf.math.abs(x - xhat), axis=[1,2,3]) / beta, axis=-1)
    elif PARAMS['observation'] == 'mse':
        error = tf.reduce_sum(tf.reduce_sum(tf.math.square(x - xhat), axis=[1,2,3]) / (2 * beta), axis=-1)
    
    # KL divergence by closed form
    '''smooth'''
    kl = tf.reduce_sum(tf.reduce_sum(smooth * (tf.math.log(smooth + 1e-8) - tf.math.log(prob + 1e-8)), axis=1))
    kl += tf.reduce_sum(tf.reduce_sum(prob * tf.math.log(prob * PARAMS['class_num'] + 1e-8), axis=1))
    kl += tf.reduce_sum(tf.reduce_sum(tf.multiply(smooth, 
                                                0.5 * (tf.reduce_sum(tf.math.pow(mean - PARAMS['prior_means'], 2) / PARAMS["sigma_vector"], axis=-1)
                                                                    - PARAMS['latent_dim']
                                                                    + tf.reduce_sum(tf.math.log(PARAMS['sigma_vector']))
                                                                    + tf.reduce_sum(tf.math.exp(logvar) / PARAMS['sigma_vector'], axis=-1)
                                                                    - tf.reduce_sum(logvar, axis=-1))), axis=-1))
    return error, kl

@tf.function
def loss_unlabeled(prob, xhat, x, mean, logvar, beta, PARAMS):
    # reconstruction error
    if PARAMS['observation'] == 'abs':
        error = tf.reduce_sum(tf.reduce_sum(tf.math.abs(x - xhat), axis=[1,2,3]) / beta, axis=-1)
    elif PARAMS['observation'] == 'mse':
        error = tf.reduce_sum(tf.reduce_sum(tf.math.square(x - xhat), axis=[1,2,3]) / (2 * beta), axis=-1)
    
    # KL divergence by closed form
    '''smooth'''
    kl = tf.reduce_sum(tf.reduce_sum(prob * tf.math.log(prob * PARAMS['class_num'] + 1e-8), axis=1))
    kl += tf.reduce_sum(tf.reduce_sum(tf.multiply(prob, 
                                                0.5 * (tf.reduce_sum(tf.math.pow(mean - PARAMS['prior_means'], 2) / PARAMS["sigma_vector"], axis=-1)
                                                                    - PARAMS['latent_dim']
                                                                    + tf.reduce_sum(tf.math.log(PARAMS['sigma_vector']))
                                                                    + tf.reduce_sum(tf.math.exp(logvar) / PARAMS['sigma_vector'], axis=-1)
                                                                    - tf.reduce_sum(logvar, axis=-1))), axis=-1))
    return error, kl
#%%
@tf.function
def sample_gumbel(shape): 
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + 1e-8) + 1e-8)

@tf.function
def gumbel_max_sample(probs): 
    y = tf.math.log(probs + 1e-8) + sample_gumbel(tf.shape(probs))
    if PARAMS['hard']:
        y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
#%%
@tf.function
def train_step(x_batch_L, x_batch_L_aug, y_batch_L, x_batch_U, x_batch_U_weak, x_batch_U_strong, beta, PARAMS):
    with tf.GradientTape(persistent=True) as tape:
        '''labeled'''
        mean, logvar = encoder(x_batch_L)
        epsilon = tf.random.normal((tf.shape(x_batch_L)[0], PARAMS['class_num'], PARAMS['latent_dim']))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        
        y_batch_L *= 1. - 0.001
        y_batch_L += 0.001 / PARAMS['class_num']
        y = gumbel_max_sample(y_batch_L)
        
        z = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        
        xhat_L = decoder(z, training=True) 
        prob_L = classifier(x_batch_L, training=True)
        
        error_L, kl_L = loss_labeled(y_batch_L, prob_L, xhat_L, x_batch_L, mean, logvar, beta, PARAMS)
        
        '''unlabeled'''
        mean, logvar = encoder(x_batch_U)
        epsilon = tf.random.normal((tf.shape(x_batch_U)[0], PARAMS['class_num'], PARAMS['latent_dim']))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        
        prob_U = classifier(x_batch_U, training=True)
        y = gumbel_max_sample(prob_U)
        
        z = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        
        xhat_U = decoder(z, training=True) 
        
        error_U, kl_U = loss_unlabeled(prob_U, xhat_U, x_batch_U, mean, logvar, beta, PARAMS)
        
        cce = tf.zeros(())
        if PARAMS['pseudo_label']:
            '''labeled'''
            prob_ = classifier(x_batch_L_aug)
            cce += - tf.reduce_mean(tf.reduce_sum(tf.multiply(y_batch_L, tf.math.log(prob_ + 1e-8)), axis=-1))    
                    
            '''unlabeled'''
            prob_ = classifier(x_batch_U_weak, training=True)
            pseudo = tf.one_hot(tf.argmax(prob_, axis=-1), depth=PARAMS['class_num'])
            indicator = tf.cast(tf.reduce_max(prob_, axis=-1) >= PARAMS['threshold'], tf.float32)
            prob_aug = classifier(x_batch_U_strong, training=True)
            cce += - tf.reduce_mean(indicator * tf.reduce_sum(tf.multiply(pseudo, tf.math.log(prob_aug + 1e-8)), axis=-1))

        # if PARAMS['optimal_interpolation']:
        
        loss = (error_L + kl_L) / x_batch_L.shape[0] + (error_U + kl_U) / x_batch_U.shape[0]
        loss += tf.cast(PARAMS['lambda1'], tf.float32) * cce
        
    grad1 = tape.gradient(loss, encoder.trainable_weights)
    opt_encoder.apply_gradients(zip(grad1, encoder.trainable_weights))
    if PARAMS['ema']:
        ema_encoder.apply(encoder.trainable_weights)
        
    grad2 = tape.gradient(loss, decoder.trainable_weights)
    opt_decoder.apply_gradients(zip(grad2, decoder.trainable_weights))
    if PARAMS['ema']:
        ema_decoder.apply(decoder.trainable_weights)
        
    grad3 = tape.gradient(loss, classifier.trainable_weights)
    opt_classifier.apply_gradients(zip(grad3, classifier.trainable_weights))
    if PARAMS['ema']:
        ema_classifier.apply(classifier.trainable_weights)
    return loss, error_L, error_U, kl_L, kl_U, cce, xhat_L, xhat_U

# @tf.function 
# def beta_step(x_batch, beta):
#     Dz = model(x_batch)[-1]
#     beta += tf.reduce_sum(tf.reduce_mean(tf.math.abs(x_batch - Dz), axis=0), axis=-1)
#     return beta
#%%
step = 0
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))
#%%
def generate_and_save_images(images, epochs):
    plt.figure(figsize=(15, 15))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        if i < 18:
            plt.imshow((images[i] + 1) / 2)
        else:
            plt.imshow((images[17 - i] + 1) / 2)
        plt.axis('off')
    plt.savefig('./assets/{}/image_at_epoch_{}.png'.format(PARAMS['data'], epochs))
    # plt.show()
    plt.close()
#%%
loss_ = []
error_ = []
kl_ = []
cce_ = []

if PARAMS['beta_trainable']:
    betapath = []
    beta = tf.Variable(PARAMS['lambda2'], trainable=True, dtype=tf.float32)
    betapath.append(beta)
else:
    beta = tf.cast(PARAMS['lambda2'], tf.float32)

# tau = PARAMS['tau']

for _ in progress_bar:
    idx = np.random.choice(np.arange(PARAMS['labeled']), PARAMS['batch_size'], replace=False)
    x_batch_L = tf.cast(np.array([(x_train_L[i] - 127.5) / 127.5 for i in idx]), tf.float32)
    x_batch_L_aug = weak_aug(x_batch_L)
    y_batch_L = np.array([y_train_onehot[i] for i in idx])
    idx = np.random.choice(np.arange(50000 - PARAMS['labeled']), PARAMS['batch_size'] * PARAMS['ratio'], replace=False)
    x_batch_U = tf.cast(np.array([(x_train_U[i] - 127.5) / 127.5 for i in idx]), tf.float32)
    x_batch_U_weak = weak_aug(x_batch_U)
    x_batch_U_strong = (np.array([np.array(strong_aug(Image.fromarray(x_train_U[i]))) for i in idx]) - 127.5) / 127.5
    
    # idx = np.random.choice(np.arange(PARAMS['labeled']), PARAMS['batch_size'], replace=False)
    # x_batch_L = np.array([np.load(data_dir + '/x_labeled_{}.npy'.format(i)) for i in idx])
    # y_batch_L = np.array([np.load(data_dir + '/y_labeled_{}.npy'.format(i)) for i in idx])
    # idx = np.random.choice(np.arange(50000 - PARAMS['labeled']), PARAMS['batch_size'], replace=False)
    # x_batch_U = np.array([np.load(data_dir + '/x_unlabeled_{}.npy'.format(i)) for i in idx])
    # x_batch_U_aug = weak_aug(x_batch_U)
    
    loss, error_L, error_U, kl_L, kl_U, cce, xhat_L, xhat_U = train_step(x_batch_L, x_batch_L_aug, y_batch_L, x_batch_U, x_batch_U_weak, x_batch_U_strong, beta, PARAMS)
        
    loss_.append(loss.numpy())
    error_.append(error_L.numpy() / x_batch_L.shape[0] + error_U.numpy() / x_batch_U.shape[0])
    kl_.append(kl_L.numpy() / x_batch_L.shape[0] + kl_U.numpy() / x_batch_U.shape[0])
    cce_.append(cce.numpy() / x_batch_U.shape[0])
    
    # if PARAMS['beta_trainable']:
    #     # beta training by Alternatning algorithm
    #     beta = 0
    #     for idx in range(20):
    #         x_batch_L = np.load(data_dir + '/x_labeled_batch{}.npy'.format(idx))
    #         beta = beta_step(x_batch_L, beta)
    #     for idx in range(372):
    #         x_batch_U = np.load(data_dir + '/x_unlabeled_batch{}.npy'.format(idx))
    #         beta = beta_step(x_batch_U, beta)
    #     beta = beta / 50000. + PARAMS['lambda1']
    #     betapath.append(beta.numpy())
    
    step += 1
    
    prob_U = classifier(x_batch_U)
    y_batch_U = np.array([np.load(data_dir + '/y_unlabeled_{}.npy'.format(i)) for i in idx])
    cls = np.sum((np.where(y_batch_U == 1)[1] - np.argmax(prob_U, axis=1)) != 0) / x_batch_U.shape[0]
    
    progress_bar.set_description('setting: batch_size={}, lr={:.5f}, lambda1={}, lambda2={}, sigma1={}, sigma2={} | iteration {}/{} | loss {:.3f}, recon {:.3f}, kl {:.3f}, cce {:.3f}, cls {:.3f}'.format(
        PARAMS['batch_size'], PARAMS['learning_rate'], PARAMS['lambda1'], PARAMS['lambda2'], PARAMS['sigma1'], PARAMS['sigma2'], 
        step, PARAMS['epochs'], 
        loss_[-1], error_[-1], kl_[-1], cce_[-1], cls)) 
    
    # tau = tf.maximum(0.5, tau * tf.math.exp(-1e-7 * step))
    
    if step % 100 == 0:
        print('setting: batch_size={}, lr={:.5f}, lambda1={}, lambda2={}, sigma1={}, sigma2={} | iteration {}/{} | loss {:.3f}, recon {:.3f}, kl {:.3f}, cls {:.3f}'.format(
            PARAMS['batch_size'], PARAMS['learning_rate'], PARAMS['lambda1'], PARAMS['lambda2'], PARAMS['sigma1'], PARAMS['sigma2'], 
            step, PARAMS['epochs'], 
            loss_[-1], error_[-1], kl_[-1], cce_[-1]))
    
    if step % 1000 == 0:
        generate_and_save_images(tf.concat([xhat_U, xhat_L], axis=0), step)
    
    if step == PARAMS['epochs']: break
#%%
asset_path = 'weights_{}_{}_shot'.format(PARAMS['data'], date_time)
#%%
encoder.save_weights('./assets/{}/{}/encoder_weights'.format(PARAMS['data'], asset_path))
decoder.save_weights('./assets/{}/{}/decoder_weights'.format(PARAMS['data'], asset_path))
classifier.save_weights('./assets/{}/{}/classifier_weights'.format(PARAMS['data'], asset_path))
#%%
# asset_path = 'weights_{}_{}_shot'.format(PARAMS['data'], '210819_1704')

# encoder = CIFAR10_shot.Encoder(PARAMS) 
# decoder = CIFAR10_shot.Decoder(PARAMS) 
# classifier = CIFAR10_shot.Classifier(PARAMS)

# encoder.load_weights('./assets/{}/{}/encoder_weights'.format(PARAMS['data'], asset_path))
# decoder.load_weights('./assets/{}/{}/decoder_weights'.format(PARAMS['data'], asset_path))
# classifier.load_weights('./assets/{}/{}/classifier_weights'.format(PARAMS['data'], asset_path))
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
'''test dataset'''
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5

from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(PARAMS['batch_size'])
#%%
mean, logvar = encoder(x_train[:PARAMS['batch_size'], ...])
epsilon = tf.random.normal((tf.shape(x_batch_U)[0], PARAMS['class_num'], PARAMS['latent_dim']))
z = mean + tf.math.exp(logvar / 2) * epsilon 

prob = classifier(x_train[:PARAMS['batch_size'], ...], training=False)
y = gumbel_max_sample(prob)

z = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)

xhat = decoder(z, training=True) 

plt.figure(figsize=(15, 15))
for i in range(PARAMS['batch_size']):
    plt.subplot(8, 8, i+1)
    plt.imshow((xhat[i] + 1) / 2)
    plt.axis('off')
plt.savefig('./assets/{}/{}/test_recon.png'.format(PARAMS['data'], asset_path),
            dpi=600, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
'''classification error''' 
error_count = 0
for x_batch, y_batch in test_dataset:
    prob = classifier(x_batch, training=False)
    error_count += len(np.where(np.squeeze(y_batch) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
classification_error = error_count / len(y_test)
print('classification error:', classification_error)
#%%
'''V-nat'''
var_list = []
for k in range(PARAMS['class_num']):
    x = x_test[np.where(y_test == k)[0]]
    mean, logvar = encoder(x, training=False)  
    var = np.exp(logvar.numpy())
    var_list.append(var[:, k, :])
var_list = np.array(var_list)

V_nat = np.log(np.mean(PARAMS['sigma_vector'] / var_list, axis=1))

k = 1
delta = 0.5
print('cardinality of activated latent subspace:', sum(V_nat[k] > delta))

plt.figure(figsize=(7, 3))
plt.bar(np.arange(PARAMS['latent_dim']), V_nat[k], width=2) 
plt.xlabel("latent dimensions", size=14)
plt.ylabel("V-nat", size=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=6)
plt.savefig('./assets/{}/{}/vnat.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%