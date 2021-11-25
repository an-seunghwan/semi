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
from datetime import datetime
import sys
import json
import os
os.chdir(r'D:\semi')

now = datetime.now() 
date_time = now.strftime("%y%m%d_%H%M%S")

from modules import CIFAR10
#%%
PARAMS = {
    "data": 'cifar10',
    "batch_size": 128,
    "labeled_batch_size": 16,
    "iterations": 100000, 
    "learning_rate1": 0.001, 
    "learning_rate2": 0.0001, 
    "data_dim": 32,
    "channel": 3, 
    "class_num": 10,
    "labeled": 5000,
    
    "z_mask": 'checkerboard',
    "c_mask": 'half',
    "z_dim": 128, 
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
    
    "ema": True,
    "slope": 0.1,
    "widen_factor": 2,
}
PARAMS['z_nf_dim'] = PARAMS['z_dim'] // 2
PARAMS['c_nf_dim'] = PARAMS['c_dim'] // 2

data_dir = r'D:\cifar10_{}'.format(PARAMS['labeled'])

with open('./assets/{}/params_{}_{}.json'.format(PARAMS['data'], PARAMS['lambda1'], PARAMS['lambda2']), 'w') as f:
    json.dump(PARAMS, f, indent=4, separators=(',', ': '))
    
asset_path = 'weights_{}'.format(date_time)
#%%
'''triaining log file'''
sys.stdout = open("./assets/{}/log_{}_{}.txt".format(PARAMS['data'], PARAMS['lambda1'], PARAMS['lambda2']), "w")

print(
'''
semi-supervised learning with flow-based model
'''
)

from pprint import pprint
pprint(PARAMS)
#%%
autoencoder = CIFAR10.AutoEncoder(PARAMS) 
prior = CIFAR10.Prior(PARAMS)
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

if PARAMS['ema']:
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
#%%
'''test dataset'''
(_, _), (x_test, y_test) = K.datasets.cifar10.load_data()
x_test = (x_test.astype('float32') - 127.5) / 127.5
from tensorflow.keras.utils import to_categorical
y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(PARAMS['batch_size'])
#%%
@tf.function
def train_step(x_batch, x_batch_L, y_batch_L, beta, PARAMS):
    with tf.GradientTape(persistent=True) as tape:
        z, c, prob, xhat = autoencoder(x_batch)
        prior_args = prior(z, c)
        
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
        
        '''supervised learning: classifier'''
        prob_L = tf.nn.softmax(autoencoder.Classifier(x_batch_L, training=True))
        cce = tf.reduce_mean(- tf.reduce_sum(tf.multiply(y_batch_L, tf.math.log(prob_L + 1e-8)), axis=-1))
        
        '''unsupervised learning: consistency training'''
        lambda_ = tf.random.uniform((), minval=0.5)
        x1, x2 = tf.split(x_batch, 2, axis=0)
        x_tilde = lambda_ * x1 + (1 - lambda_) * x2
        epsilon = prior.cNF(c)[0]
        e1, e2 = tf.split(epsilon, 2, axis=0)
        e_tilde = lambda_ * e1 + (1 - lambda_) * e2
        c_tilde = prior.cNF.inverse(e_tilde)
        
        prob1 = tf.nn.softmax(autoencoder.Classifier(x_tilde))
        prob2 = tf.nn.softmax(c_tilde)
        kl = tf.reduce_mean(tf.reduce_sum(prob1 * (tf.math.log(prob1 + 1e-8) - tf.math.log(prob2 + 1e-8)), axis=1))
        
        loss = recon_loss / PARAMS['beta'] + PARAMS['lambda1'] * cce + PARAMS['lambda2'] * kl
        
    grad = tape.gradient(loss, autoencoder.trainable_weights)
    optimizer.apply_gradients(zip(grad, autoencoder.trainable_weights))
    if PARAMS['ema']:
        ema.apply(autoencoder.trainable_weights)
    
    grad = tape.gradient(prior_loss, prior.trainable_weights)
    optimizer_NF.apply_gradients(zip(grad, prior.trainable_weights))
    return [[loss, recon_loss, z_prior_loss, c_prior_loss, cce, kl], 
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
        xhat = autoencoder.Decoder(z, label, training=False)
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

test_error = []

'''test classification error''' 
error_count = 0
for x_batch, y_batch in test_dataset:
    prob = autoencoder.Classifier(x_batch, training=False)
    error_count += len(np.where(np.squeeze(y_batch) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
test_error.append(error_count / len(y_test))

for _ in progress_bar:
    # label 
    idx = np.random.choice(np.arange(PARAMS['labeled']), PARAMS['labeled_batch_size'], replace=False)
    x_batch_L = np.array([np.load(data_dir + '/x_labeled_{}.npy'.format(i)) for i in idx])
    x_batch_L = (tf.cast(x_batch_L, tf.float32) - 127.5) / 127.5
    y_batch_L = np.array([np.load(data_dir + '/y_labeled_{}.npy'.format(i)) for i in idx])
    # label and unlabel
    idx = np.random.choice(np.arange(50000), PARAMS['batch_size'], replace=False)
    x_batch = np.array([np.load(data_dir + '/x_{}.npy'.format(i)) for i in idx])
    x_batch = (tf.cast(x_batch, tf.float32) - 127.5) / 127.5
    
    losses, outputs = train_step(x_batch, y_batch, PARAMS) 
    
    step += 1
        
    progress_bar.set_description('iteration {}/{} | loss {:.3f}, recon {:.3f}, z prior {:.3f}, c prior {:.3f}, cls {:.3f}, kl {:.3f}, test {:.3f}'.format(
        step, PARAMS['iterations'], 
        losses[0].numpy(), losses[1].numpy(), losses[2].numpy(), losses[3].numpy(), losses[4].numpy(), losses[5].numpy(), test_error[-1]
    )) 
    
    step += 1
    
    if step % 5000 == 0:
        '''test classification error''' 
        error_count = 0
        for x_batch, y_batch in test_dataset:
            prob = autoencoder.Classifier(x_batch, training=False)
            error_count += len(np.where(np.squeeze(y_batch) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
        test_error.append(error_count / len(y_test))
        
        print('iteration {}/{} | loss {:.3f}, recon {:.3f}, z prior {:.3f}, c prior {:.3f}, cls {:.3f}, kl {:.3f}, test {:.3f}'.format(
            step, PARAMS['iterations'], 
            losses[0].numpy(), losses[1].numpy(), losses[2].numpy(), losses[3].numpy(), losses[4].numpy(), losses[5].numpy(), test_error[-1]
        ))
        
    if step % 10000 == 0:
        x = np.load(data_dir + '/x_{}.npy'.format(0))
        x = (tf.cast(x, tf.float32) - 127.5) / 127.5
        generate_and_save_images(x, step)
    
    if step == PARAMS['iterations']: break
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
'''test classification error phase'''
plt.rc('xtick', labelsize=10)   
plt.rc('ytick', labelsize=10)   
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(test_error, label='test classification error')
leg = ax.legend(fontsize=15, loc='upper right')
plt.savefig('./assets/{}/{}/classification_error_phase.png'.format(PARAMS['data'], asset_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%