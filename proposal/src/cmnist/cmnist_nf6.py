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
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import json
from datetime import datetime
import sys
import os
os.chdir(r'D:\normalizing_flow')
# os.chdir('/home1/prof/jeon/an/normalizing_flow')

data_dir = r'D:\cmnist'
# data_dir = '/home1/prof/jeon/an/cmnist'

now = datetime.now() 
date_time = now.strftime("%y%m%d_%H%M%S")
#%%
PARAMS = {
    "batch_size": 256,
    "iterations": 20000, 
    "learning_rate1": 0.001, 
    "learning_rate2": 0.0001, 
    "data": "cmnist",
    "data_dim": 32,
    "channel": 3, 
    "class_num": 10,
    
    "z_mask": 'checkerboard',
    "c_mask": 'half',
    "z_dim": 16, 
    "c_dim": 10,
    "z_embedding_dim": 256,
    "c_embedding_dim": 256,
    "K1": 8,
    "K2": 8,
    "coupling_MLP_num": 4,
    
    "reg": 0.01,
    "BN_in_NF": False,
    "decay_steps": 500,
    "decay_rate": 0.95,
    "gradclip": 1.,
    
    "beta": 1.,
    "lambda1": 5, 
    "lambda2": 10, 
    "activation": 'tanh',
    "observation": 'mse',
}

# PARAMS['z_nf_dim'] = [PARAMS['z_dim'] // 2, PARAMS['z_dim'] // 2]
# PARAMS['c_nf_dim'] = [PARAMS['c_dim'] // 2, PARAMS['c_dim'] // 2]
PARAMS['z_nf_dim'] = PARAMS['z_dim'] // 2
PARAMS['c_nf_dim'] = PARAMS['c_dim'] // 2

with open('./assets/{}/params_{}.json'.format(PARAMS['data'], date_time), 'w') as f:
    json.dump(PARAMS, f, indent=4, separators=(',', ': '))
    
asset_path = 'weights_{}'.format(date_time)
#%%
'''training log file'''
sys.stdout = open("./assets/{}/log6_{}.txt".format(PARAMS['data'], date_time), "w")

print(
'''
1. conditional probability modeling p(x|y)
2. conditional normalizing flow
3. additional classification error
4. mutual information regularization (reconstruction)
5. learning rate decay on Normalizing Flow
6. Batch Normalization in Normalizing Flow
'''
)

pprint(PARAMS)
print('\n')
#%%
# if PARAMS['BN_in_NF']:
from modules import CMNIST6
autoencoder = CMNIST6.AutoEncoder(PARAMS)
prior = CMNIST6.Prior(PARAMS)
prior.build_graph()
    
optimizer = K.optimizers.Adam(PARAMS["learning_rate1"])

if PARAMS['gradclip'] is None:
    if PARAMS['decay_steps'] is None:
        optimizer_NF = K.optimizers.Adam(PARAMS['learning_rate2']) 
    else:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=PARAMS["learning_rate2"], decay_steps=PARAMS['decay_steps'], decay_rate=PARAMS['decay_rate'])
        optimizer_NF = K.optimizers.Adam(learning_rate) 
else:
    if PARAMS['decay_steps'] is None:
        optimizer_NF = K.optimizers.Adam(PARAMS['learning_rate2'], clipvalue=PARAMS['gradclip'])
    else:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=PARAMS["learning_rate2"], decay_steps=PARAMS['decay_steps'], decay_rate=PARAMS['decay_rate'])
        optimizer_NF = K.optimizers.Adam(learning_rate, clipvalue=PARAMS['gradclip']) 
#%%
@tf.function
def train_step(x_batch, y_batch, PARAMS):
    
    with tf.GradientTape(persistent=True) as tape:
        z, c, prob, xhat = autoencoder(x_batch, training=True)
        prior_args = prior(z, c, y_batch)
        
        '''reconstruction'''
        if PARAMS['observation'] == 'mse':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(xhat - x_batch) / 2., axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'abs':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(xhat - x_batch), axis=[1, 2, 3]))
        else:
            assert 0, "Unsupported observation model: {}".format(PARAMS['observation'])
            
        '''prior'''
        z_prior_loss = tf.reduce_mean(tf.reduce_sum(tf.square(prior_args[0] - 0) / 2., axis=1))
        z_prior_loss -= tf.reduce_mean(prior_args[1], axis=-1)
        c_prior_loss = tf.reduce_mean(tf.reduce_sum(tf.square(prior_args[2] - 0) / 2., axis=1))
        c_prior_loss -= tf.reduce_mean(prior_args[3], axis=-1)
        prior_loss = z_prior_loss + c_prior_loss
        
        '''classification'''
        classification_loss = tf.reduce_mean(- tf.reduce_sum(y_batch * tf.math.log(prob + 1e-8), axis=-1))
        
        info = tf.zeros(())
        '''mutual information'''
        if PARAMS['lambda2']:
            c_recon = autoencoder.cEncoder(xhat, training=True)
            prob_recon = tf.nn.softmax(c_recon, axis=-1)
            info += tf.reduce_mean(- tf.reduce_sum(y_batch * tf.math.log(prob_recon + 1e-8), axis=-1))
        
        loss = recon_loss / PARAMS['beta'] + PARAMS['lambda1'] * classification_loss + PARAMS['lambda2'] * info
        
    grad = tape.gradient(loss, autoencoder.trainable_weights)
    optimizer.apply_gradients(zip(grad, autoencoder.trainable_weights))
    
    grad = tape.gradient(prior_loss, prior.trainable_weights)
    optimizer_NF.apply_gradients(zip(grad, prior.trainable_weights))
    
    return [[loss, recon_loss, z_prior_loss, c_prior_loss, classification_loss, info], 
            [z, c, xhat] + prior_args]
#%%
def generate_and_save_images(x, epochs):
    z = autoencoder.zEncoder(x, training=False)
    
    plt.figure(figsize=(10, 2))
    plt.subplot(1, PARAMS['class_num']+1, 1)
    plt.imshow((x[0] + 1) / 2)
    plt.title('original')
    plt.axis('off')
    for i in range(PARAMS['class_num']):
        label = np.zeros((z.shape[0], PARAMS['class_num']))
        label[:, i] = 1
        xhat = autoencoder.Decoder([z, label], training=False)
        plt.subplot(1, PARAMS['class_num']+1, i+2)
        plt.imshow((xhat[0] + 1) / 2)
        plt.title('{}'.format(i))
        plt.axis('off')
    plt.savefig('./assets/{}/image_at_epoch_{}.png'.format(PARAMS['data'], epochs))
    # plt.show()
    plt.close()
#%%
step = 0
progress_bar = tqdm(range(PARAMS['iterations']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['iterations']))

for _ in progress_bar:
    idx = np.random.choice(np.arange(60000), PARAMS['batch_size'], replace=False)
    x_batch = np.array([np.load(data_dir + '/train/x_{}.npy'.format(i)) for i in idx])
    x_batch = tf.cast(x_batch, tf.float32)
    y_batch = np.array([np.load(data_dir + '/train/y_{}.npy'.format(i)) for i in idx])
    
    losses, outputs = train_step(x_batch, y_batch, PARAMS) 
    
    step += 1
    
    progress_bar.set_description('iteration {}/{} | loss {:.3f}, recon {:.3f}, z prior {:.3f}, c prior {:.3f}, cls {:.3f}, info {:.3f}'.format(
        step, PARAMS['iterations'], 
        losses[0].numpy(), losses[1].numpy(), losses[2].numpy(), losses[3].numpy(), losses[4].numpy(), losses[5].numpy()
    )) 
    
    if step % 1000 == 0:
        x = np.load(data_dir + '/train/x_{}.npy'.format(46))[None, ...]
        generate_and_save_images(x, step)
    
        print('iteration {}/{} | loss {:.3f}, recon {:.3f}, z prior {:.3f}, c prior {:.3f}, cls {:.3f}, info {:.3f}'.format(
            step, PARAMS['iterations'], 
            losses[0].numpy(), losses[1].numpy(), losses[2].numpy(), losses[3].numpy(), losses[4].numpy(), losses[5].numpy()
        ))

    # if step == PARAMS['iterations']: break
#%%
autoencoder.zEncoder.save_weights('./assets/{}/{}/weights_zEncoder/weights'.format(PARAMS['data'], asset_path))
autoencoder.cEncoder.save_weights('./assets/{}/{}/weights_cEncoder/weights'.format(PARAMS['data'], asset_path))
autoencoder.Decoder.save_weights('./assets/{}/{}/weights_Decoder/weights'.format(PARAMS['data'], asset_path))

autoencoder.summary()
print('\n')
autoencoder.zEncoder.summary()
print('\n')
autoencoder.cEncoder.summary()
print('\n')
autoencoder.Decoder.summary()
print('\n')

prior.save_weights('./assets/{}/{}/weights_Prior/weights'.format(PARAMS['data'], asset_path))
prior.summary()
#%%