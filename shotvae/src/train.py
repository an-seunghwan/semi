#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
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
from pprint import pprint
from datetime import datetime
import sys
import json
import os
os.chdir(r'D:\semi\shotvae')

now = datetime.now() 
date_time = now.strftime("%y%m%d_%H%M%S")

from modules import CIFAR10
#%%
PARAMS = {
    "data": 'cifar10',
    "batch_size": 512,
    "epochs": 600,
    "data_dim": 32,
    "channel": 3, 
    "class_num": 10,
    "temperature": 0.67,
    "latent_dim": 128,
    "sigma": 1.,
    "annotated_ratio": 0.1,
    
    "kbmc": 1e-3, # kl beta max continuous
    "kbmd": 1e-3, # kl beta max discrete
    "akb": 200, # the max epoch to adjust kl beta
    "ewm": 1e-3, # elbo weight max
    "aew": 400, # the epoch to adjust elbo weight to max
    "pwm": 1, # posterior weight max
    "apw": 200, # adjust posterior weight
    "wrd": 1., # the max weight for the optimal transport estimation of discrete variable 
    "wmf": 0.4, # the weight factor: epoch to adjust the weight for the optimal transport estimation of discrete variable to max
    "dmi": 2.3, # threshold of discrete kl-divergence
    
    "learning_rate": 0.1, 
    "beta_1": 0.9, # beta_1 in SGD or Adam
    "adjust_lr": [400, 500, 550], # the milestone list for adjust learning rate
    "weight_decay": 5e-4 / 2, 
    "epsilon": 0.1, # the label smoothing epsilon for labeled dataset
    "activation": 'sigmoid',
    "observation": 'bce',
    
    "hard": False,
    "slope": 0.01, # pytorch default
    "widen_factor": 2,
    "depth": 28,
}

asset_path = 'weights_{}'.format(date_time)
#%%
'''triaining log file'''
sys.stdout = open("./assets/{}/log_{}.txt".format(PARAMS['data'], date_time), "w")

print(
'''
reproduce of SHOT-VAE
'''
)
#%%
model = CIFAR10.VAE(PARAMS) 
#%%
def weight_schedule(epoch, epochs, weight_max):
    return weight_max * tf.math.exp(-5 * (1 - min(1, epoch/epochs)) ** 2)
#%%
'''dataset'''
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

if PARAMS['activation'] == 'tanh':
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_test = (x_test.astype('float32') - 127.5) / 127.5
elif PARAMS['activation'] == 'sigmoid':
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
else:
    assert 0, "Unsupported observation model: {}".format(PARAMS['observation'])
y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])

np.random.seed(1)
'''ensure that all classes are balanced '''
lidx = np.concatenate([np.random.choice(np.where(y_train == i)[0], 
                                        int((len(x_train) * PARAMS['annotated_ratio']) / PARAMS['class_num']), 
                                        replace=False) 
                        for i in range(PARAMS['class_num'])])
x_train_L = x_train[lidx]
y_train_L = y_train_onehot[lidx]

train_L_dataset = tf.data.Dataset.from_tensor_slices((x_train_L, y_train_L)).shuffle(len(x_train_L), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(PARAMS['batch_size'])

PARAMS['iterations'] = len(train_dataset)
#%%
with open('./assets/{}/params_{}.json'.format(PARAMS['data'], date_time), 'w') as f:
    json.dump(PARAMS, f, indent=4, separators=(',', ': '))
pprint(PARAMS)
#%%
def augmentation(image):
    paddings = tf.constant([[0, 0],
                            [4, 4],
                            [4, 4],
                            [0, 0]])
    image = tf.pad(image, paddings, 'REFLECT')
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (image.shape[0], 32, 32, image.shape[-1]))
    return image
#%%
def gaussian_kl_divergence(mean1, mean2, log_sigma1, log_sigma2):
    return 0.5 * tf.reduce_sum(tf.math.square(mean1 - mean2) / tf.math.exp(2. * log_sigma2)
                               + tf.math.exp(2. * log_sigma1) / tf.math.exp(2. * log_sigma2) 
                               - 1 
                               + 2. * log_sigma2 
                               - 2. * log_sigma1, axis=-1)
#%%
def test_cls_error(model, test_dataset):
    '''test classification error''' 
    error_count = 0
    for x_batch, y_batch in test_dataset:
        _, _, log_prob, _, _, _ = model(x_batch, training=False)
        error_count += len(np.where(np.squeeze(y_batch) - np.argmax(log_prob.numpy(), axis=-1) != 0)[0])
    return error_count / len(y_test)
#%%
# @tf.function
def supervised_train_step(x_batch_L, y_batch_L, PARAMS,
                            ew, kl_beta_z, kl_beta_y, pwm, mix_weight,
                            optimizer):
    eps = 1e-8
    
    with tf.GradientTape(persistent=True) as tape:
    
        mean, log_sigma, log_prob, z, y, xhat = model(x_batch_L, y_batch_L)
        
        '''reconstruction'''
        if PARAMS['observation'] == 'mse':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(xhat - x_batch_L) / 2., axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'abs':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(xhat - x_batch_L), axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'bce':
            recon_loss = tf.reduce_mean(- tf.reduce_sum(x_batch_L * tf.math.log(xhat + eps) + 
                                                        (1. - x_batch_L) * tf.math.log(1. - xhat + eps), axis=[1, 2, 3]))
        else:
            assert 0, "Unsupported observation model: {}".format(PARAMS['observation'])
            
        '''prior: KL-divergence'''
        kl_z = tf.reduce_mean(tf.reduce_sum(0.5 * (tf.math.square(mean) / PARAMS['sigma'] 
                                                    + tf.math.exp(2 * log_sigma) / PARAMS['sigma'] 
                                                    + tf.math.log(PARAMS['sigma'])
                                                    - 2. * log_sigma
                                                    - 1.), axis=-1))
        kl_y = tf.reduce_mean(tf.reduce_sum(tf.math.exp(log_prob) * (log_prob - tf.math.log(1. / PARAMS['class_num'])), axis=1))
        elbo_loss_L = recon_loss + kl_beta_z * kl_z + kl_beta_y * tf.math.abs(kl_y - PARAMS['dmi'])
        
        '''mix-up'''
        # no-gradient error!!!
        # x_batch_L_shuffle = tf.random.shuffle(x_batch_L)
        # y_batch_L_shuffle = tf.random.shuffle(y_batch_L)
        # mean_shuffle = tf.random.shuffle(mean)
        # logvar_shuffle = tf.random.shuffle(logvar)
        x_batch_L_shuffle = tf.gather(x_batch_L, tf.random.shuffle(tf.range(x_batch_L.shape[0])))
        y_batch_L_shuffle = tf.gather(y_batch_L, tf.random.shuffle(tf.range(y_batch_L.shape[0])))
        mean_shuffle = tf.gather(mean, tf.random.shuffle(tf.range(mean.shape[0])))
        log_sigma_shuffle = tf.gather(log_sigma, tf.random.shuffle(tf.range(log_sigma.shape[0])))
        
        x_batch_L_mix = mix_weight * x_batch_L_shuffle + (1. - mix_weight) * x_batch_L
        mean_mix = mix_weight * mean_shuffle + (1. - mix_weight) * mean
        sigma_mix = mix_weight * tf.math.exp(log_sigma_shuffle) + (1. - mix_weight) * tf.math.exp(log_sigma)
        smoothed_mean_mix, smoothed_log_sigma_mix, smoothed_log_prob_mix, _, _, _ = model(x_batch_L_mix)
        
        posterior_loss_z = tf.reduce_mean(tf.math.square(smoothed_mean_mix - mean_mix))
        posterior_loss_z += tf.reduce_mean(tf.math.square(tf.math.exp(smoothed_log_sigma_mix) - sigma_mix))
        posterior_loss_y = - tf.reduce_mean(mix_weight * tf.reduce_sum(y_batch_L_shuffle * smoothed_log_prob_mix, axis=-1))
        posterior_loss_y += - tf.reduce_mean((1. - mix_weight) * tf.reduce_sum(y_batch_L * smoothed_log_prob_mix, axis=-1))
        
        elbo_loss_L += kl_beta_z * pwm * posterior_loss_z
        loss_supervised = ew * elbo_loss_L + posterior_loss_y
            
    grad = tape.gradient(loss_supervised, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    
    return [[loss_supervised, recon_loss, kl_z, kl_y, posterior_loss_z, posterior_loss_y], 
            [mean, log_sigma, log_prob, z, y, xhat]]
#%%
# @tf.function
def unsupervised_train_step(x_batch, x_batch_shuffle, mean_shuffle, log_sigma_shuffle, log_prob_shuffle, PARAMS,
                            ew, kl_beta_z, kl_beta_y, pwm, ucw, mix_weight,
                            optimizer):
    eps = 1e-8
    
    with tf.GradientTape(persistent=True) as tape:
        
        mean, log_sigma, log_prob, z, y, xhat = model(x_batch)
        
        '''reconstruction'''
        if PARAMS['observation'] == 'mse':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(xhat - x_batch) / 2., axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'abs':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(xhat - x_batch), axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'bce':
            recon_loss = tf.reduce_mean(- tf.reduce_sum(x_batch * tf.math.log(xhat + eps) + 
                                                        (1. - x_batch) * tf.math.log(1. - xhat + eps), axis=[1, 2, 3]))
        else:
            assert 0, "Unsupported observation model: {}".format(PARAMS['observation'])
            
        '''prior: KL-divergence'''
        kl_z = tf.reduce_mean(tf.reduce_sum(0.5 * (tf.math.square(mean) / PARAMS['sigma'] 
                                                    + tf.math.exp(2 * log_sigma) / PARAMS['sigma'] 
                                                    + tf.math.log(PARAMS['sigma'])
                                                    - 2. * log_sigma
                                                    - 1.), axis=-1))
        kl_y = tf.reduce_mean(tf.reduce_sum(tf.math.exp(log_prob) * (log_prob - tf.math.log(1. / PARAMS['class_num'])), axis=1))
        elbo_loss_U = recon_loss + kl_beta_z * kl_z + kl_beta_y * tf.math.abs(kl_y - PARAMS['dmi'])
        
        '''mix-up'''
        x_batch_mix = mix_weight * x_batch_shuffle + (1. - mix_weight) * x_batch
        mean_mix = mix_weight * mean_shuffle + (1. - mix_weight) * mean
        sigma_mix = mix_weight * tf.math.exp(log_sigma_shuffle) + (1 - mix_weight) * tf.math.exp(log_sigma)
        pseudo_label = mix_weight * tf.math.exp(log_prob_shuffle) + (1. - mix_weight) * tf.math.exp(log_prob)
        smoothed_mean_mix, smoothed_log_sigma_mix, smoothed_log_prob_mix, _, _, _ = model(x_batch_mix)
        
        posterior_loss_z = tf.reduce_mean(tf.math.square(smoothed_mean_mix - mean_mix))
        posterior_loss_z += tf.reduce_mean(tf.math.square(tf.math.exp(smoothed_log_sigma_mix) - sigma_mix))
        posterior_loss_y = - tf.reduce_mean(tf.reduce_sum(pseudo_label * smoothed_log_prob_mix, axis=-1))
        
        elbo_loss_U += kl_beta_z * pwm * posterior_loss_z
        loss_unsupervised = ew * elbo_loss_U + ucw * posterior_loss_y
            
    grad = tape.gradient(loss_unsupervised, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    
    return [[loss_unsupervised, recon_loss, kl_z, kl_y, posterior_loss_z, posterior_loss_y], 
            [mean, log_sigma, log_prob, z, y, xhat]]
#%%
# def generate_and_save_images(x, epochs):
#     z = autoencoder.zEncoder(x, training=False)
    
#     plt.figure(figsize=(10, 2))
#     plt.subplot(1, PARAMS['class_num']+1, 1)
#     plt.imshow((x[0] + 1) / 2)
#     plt.title('original')
#     plt.axis('off')
#     for i in range(PARAMS['class_num']):
#         label = np.zeros((z.shape[0], PARAMS['class_num']))
#         label[:, i] = 1
#         xhat = autoencoder.Decoder(z, label, training=False)
#         plt.subplot(1, PARAMS['class_num']+1, i+2)
#         plt.imshow((xhat[0] + 1) / 2)
#         plt.title('{}'.format(i))
#         plt.axis('off')
#     plt.savefig('./assets/{}/image_at_epoch_{}.png'.format(PARAMS['data'], epochs))
#     # plt.show()
#     plt.close()

def generate_and_save_images(xhat, epoch):
    xhat = xhat.numpy()
    
    plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow((xhat[i] + 1) / 2)
        plt.axis('off')
    plt.savefig('./assets/{}/image_at_epoch_{}.png'.format(PARAMS['data'], epoch))
    # plt.show()
    plt.close()
#%%
test_error = [test_cls_error(model, test_dataset)] # initial test classification error

learning_rate_fn = K.optimizers.schedules.PiecewiseConstantDecay(
    PARAMS['adjust_lr'], [PARAMS['learning_rate'] * t for t in [1., 0.1, 0.01, 0.001]]
)

for epoch in range(PARAMS['epochs']):
    
    '''warm-up'''
    if epoch == 0:
        optimizer = K.optimizers.SGD(learning_rate=PARAMS['learning_rate'] * 0.2,
                                    momentum=PARAMS['beta_1'])
        # optimizer = K.optimizers.Adam(learning_rate=PARAMS['learning_rate'] * 0.2,
        #                             beta_1=PARAMS['beta_1'])
    else:
        optimizer.lr.assign(learning_rate_fn(epoch))

    '''weights of loss terms'''
    # elbo part weight
    ew = weight_schedule(epoch, PARAMS['epochs'], PARAMS['ewm'])
    # kl-divergence weight
    kl_beta_z = weight_schedule(epoch, PARAMS['epochs'], PARAMS['kbmc'])
    kl_beta_y = weight_schedule(epoch, PARAMS['epochs'], PARAMS['kbmd'])
    # unsupervised classification weight
    pwm = weight_schedule(epoch, PARAMS['epochs'], PARAMS['pwm'])
    # optimal transport weight
    ucw = weight_schedule(epoch, round(PARAMS['wmf'] * PARAMS['epochs']), PARAMS['wrd'])
        
    step = 0
    progress_bar = tqdm(range(PARAMS['iterations']))
    progress_bar.set_description('iterations {}/{} | current loss ?'.format(step, PARAMS['iterations']))
    
    for _ in progress_bar:
    
        '''mini-batch'''
        x_batch_L, y_batch_L = next(iter(train_L_dataset))
        x_batch = next(iter(train_dataset))
        
        '''augmentation'''
        x_batch_L = augmentation(x_batch_L)
        x_batch = augmentation(x_batch)

        '''mix-up weight'''
        mix_weight = [tf.constant(np.random.beta(PARAMS['epsilon'], PARAMS['epsilon'])), # labeled
                      tf.constant(np.random.beta(2.0, 2.0))] # unlabeled
        
        '''labeled dataset training'''
        supervised_losses, supervised_outputs = supervised_train_step(x_batch_L, y_batch_L, PARAMS,
                                                ew, kl_beta_z, kl_beta_y, pwm, mix_weight[0], optimizer) 
        
        '''mix-up: optimal match'''
        mean, log_sigma, log_prob, _, _, _ = model(x_batch)
        mean = mean.numpy()
        log_sigma = log_sigma.numpy()
        kl_metric = np.zeros((x_batch.shape[0], x_batch.shape[0]))
        for i in range(x_batch.shape[0]):
            kl_metric[i, :] = gaussian_kl_divergence(mean, mean[i],
                                                    log_sigma, log_sigma[i]).numpy()
        unsupervised_mix_up_index = np.argsort(kl_metric, axis=1)[:, 1]
        x_batch_shuffle = tf.gather(x_batch, unsupervised_mix_up_index)
        mean_shuffle = tf.gather(mean, unsupervised_mix_up_index)
        log_sigma_shuffle = tf.gather(log_sigma, unsupervised_mix_up_index)
        log_prob_shuffle = tf.gather(log_prob, unsupervised_mix_up_index)
        
        '''unlabeled dataset training'''
        unsupervised_losses, unsupervised_outputs = unsupervised_train_step(x_batch, x_batch_shuffle, mean_shuffle, log_sigma_shuffle, log_prob_shuffle, PARAMS,
                                                                            ew, kl_beta_z, kl_beta_y, pwm, ucw, mix_weight[1], optimizer)
        
        step += 1
        
        progress_bar.set_description('epoch: {} | iteration {}/{} | supervised {:.3f}, unsupervised {:.3f}, test {:.3f}'.format(
            epoch, step, PARAMS['iterations'], 
            supervised_losses[0].numpy(), unsupervised_losses[0].numpy(), test_error[-1]
        )) 
        
        if step % 10 == 0:
            print('epoch: {} | iteration {}/{} | supervised {:.3f}, unsupervised {:.3f}, test {:.3f}'.format(
                epoch, step, PARAMS['iterations'], 
                supervised_losses[0].numpy(), unsupervised_losses[0].numpy(), test_error[-1]
            ))
        
    test_error.append(test_cls_error(model, test_dataset))
        
    if epoch % 50 == 0:
        generate_and_save_images(unsupervised_outputs[-1], epoch)
#%%
'''save model'''
# model.save_weights('./assets/{}/{}/weights'.format(PARAMS['data'], asset_path))
os.makedirs('./assets/{}/{}'.format(PARAMS['data'], asset_path))
model.save_weights('./assets/{}/{}/model.h5'.format(PARAMS['data'], asset_path), save_format="h5")
model.summary()
#%%
'''import model'''
# imported = CIFAR10.VAE(PARAMS)
# dummy = tf.random.normal((1, 32, 32, 3))
# '''call the model first'''
# _ = imported(dummy)
# imported.load_weights('./assets/{}/{}/model.h5'.format(PARAMS['data'], asset_path))
# imported(x_batch_L)
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