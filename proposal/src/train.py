#%%
import tensorflow as tf
import tensorflow.keras as K
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
from datetime import datetime
from pprint import pprint
import sys
import json
import os
os.chdir(r'D:\semi\proposal')

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
    "latent_dim": 128,
    "annotated_ratio": 0.1,
    
    "ewm": 1e-3, # elbo weight max
    "aew": 400, # the epoch to adjust elbo weight to max
    "pwm": 1, # posterior weight max
    "apw": 200, # adjust posterior weight
    "wrd": 1., # the max weight for the optimal transport estimation of discrete variable 
    "wmf": 0.4, # the weight factor: epoch to adjust the weight for the optimal transport estimation of discrete variable to max
    
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
    "decay_steps": 20,
    "decay_rate": 0.95,
    "gradclip": 1.,
    "BN_in_NF": False,
    
    # "beta": 1.,
    # "lambda1": 10, 
    # "lambda2": 10, 
    "learning_rate1": 0.1, 
    "learning_rate2": 0.0001,
    "beta_1": 0.9, # beta_1 in SGD or Adam
    "adjust_lr": [400, 500, 550], # the milestone list for adjust learning rate
    "weight_decay": 5e-4 / 2, 
    "epsilon": 0.1, # the label smoothing epsilon for labeled dataset
    "activation": 'sigmoid',
    "observation": 'bce',
    "ema": True,
    
    "slope": 0.01, # pytorch default
    "widen_factor": 2,
    "depth": 28,
}
PARAMS['z_nf_dim'] = PARAMS['z_dim'] // 2
PARAMS['c_nf_dim'] = PARAMS['c_dim'] // 2

asset_path = 'weights_{}'.format(date_time)
#%%
'''triaining log file'''
sys.stdout = open("./assets/{}/log_{}.txt".format(PARAMS['data'], date_time), "w")

print(
'''
semi-supervised learning with flow-based model
'''
)
#%%
model = CIFAR10.DeterministicVAE(PARAMS) 
model.Prior.build_graph()
if PARAMS['ema']:
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
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
def test_cls_error(model, test_dataset):
    '''test classification error''' 
    error_count = 0
    for x_batch, y_batch in test_dataset:
        prob = model.AE.get_prob(x_batch, training=False)
        error_count += len(np.where(np.squeeze(y_batch) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
    return error_count / len(y_test)
#%%
# @tf.function
def supervised_train_step(x_batch_L, y_batch_L, PARAMS,
                          mix_weight, ew, optimizer):
    eps = 1e-8
    
    with tf.GradientTape(persistent=True) as tape:
        
        [[z, c, prob, xhat], prior_args] = model(x_batch_L)
        
        '''reconstruction'''
        if PARAMS['observation'] == 'mse':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(xhat - x_batch_L) / 2., axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'abs':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(xhat - x_batch_L), axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'bce':
            recon_loss = tf.reduce_mean(- tf.reduce_sum(x_batch_L * tf.math.log(xhat + eps) + 
                                                        (1. - x_batch_L) * tf.math.log(1. - xhat + eps), axis=[1, 2, 3]))
        else:
            assert 0, "Unsupported observation model: {}".format(PARAMS['observation'])
        
        '''prior'''
        z_prior_loss = tf.reduce_mean(tf.reduce_sum(tf.square(prior_args[0] - 0) / 2., axis=1))
        z_prior_loss -= tf.reduce_mean(prior_args[1], axis=-1)
        c_prior_loss = tf.reduce_mean(tf.reduce_sum(tf.square(prior_args[2] - 0) / 2., axis=1))
        c_prior_loss -= tf.reduce_mean(prior_args[3], axis=-1)
        prior_loss = z_prior_loss + c_prior_loss
        
        '''supervised learning: classifier'''
        classification_loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(y_batch_L, tf.math.log(prob + eps)), axis=-1))
        
        '''mutual information: reconstruction'''
        recon_prob = model.AE.get_prob(xhat)
        classification_loss += - tf.reduce_mean(tf.reduce_sum(tf.multiply(y_batch_L, tf.math.log(recon_prob + eps)), axis=-1))
        
        '''mix-up'''
        x_batch_L_shuffle = tf.gather(x_batch_L, tf.random.shuffle(tf.range(x_batch_L.shape[0])))
        y_batch_L_shuffle = tf.gather(y_batch_L, tf.random.shuffle(tf.range(y_batch_L.shape[0])))
        x_batch_L_mix = mix_weight * x_batch_L_shuffle + (1. - mix_weight) * x_batch_L
        smoothed_prob_mix = model.AE.get_prob(x_batch_L_mix)
        posterior_loss_y = - tf.reduce_mean(mix_weight * tf.reduce_sum(y_batch_L_shuffle * tf.math.log(smoothed_prob_mix + eps), axis=-1))
        posterior_loss_y += - tf.reduce_mean((1. - mix_weight) * tf.reduce_sum(y_batch_L * tf.math.log(smoothed_prob_mix + eps), axis=-1))
        
        loss_supervised = ew * (recon_loss + classification_loss) + posterior_loss_y
        
    grad = tape.gradient(loss_supervised, model.AE.trainable_weights)
    optimizer[0].apply_gradients(zip(grad, model.AE.trainable_weights))
    if PARAMS['ema']:
        ema.apply(model.AE.trainable_weights)
    
    grad = tape.gradient(prior_loss, model.Prior.trainable_weights)
    optimizer[1].apply_gradients(zip(grad, model.Prior.trainable_weights))
    
    return [[loss_supervised, recon_loss, z_prior_loss, c_prior_loss, classification_loss, posterior_loss_y], 
            [z, c, xhat] + prior_args]
#%%
# @tf.function
def unsupervised_train_step(x_batch, unsupervised_mix_up_index, PARAMS,
                            mix_weight, ew, ucw, optimizer):
    eps = 1e-8
    
    with tf.GradientTape(persistent=True) as tape:
        
        [[z, c, prob, xhat], prior_args] = model(x_batch)
        
        '''reconstruction'''
        if PARAMS['observation'] == 'mse':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(xhat - x_batch) / 2., axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'abs':
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(xhat - x_batch), axis=[1, 2, 3]))
        elif PARAMS['observation'] == 'bce':
            recon_loss = tf.reduce_mean(- tf.reduce_sum(x_batch * tf.math.log(xhat + eps) + 
                                                        (1. - x_batch) * tf.math.log(1. - xhat + eps), axis=[1, 2, 3]))
        else:
            assert 0, "Unsupported observation model: {}".format(PARAMS['observation'])
        
        '''prior'''
        z_prior_loss = tf.reduce_mean(tf.reduce_sum(tf.square(prior_args[0] - 0) / 2., axis=1))
        z_prior_loss -= tf.reduce_mean(prior_args[1], axis=-1)
        c_prior_loss = tf.reduce_mean(tf.reduce_sum(tf.square(prior_args[2] - 0) / 2., axis=1))
        c_prior_loss -= tf.reduce_mean(prior_args[3], axis=-1)
        prior_loss = z_prior_loss + c_prior_loss
        
        '''mix up'''
        x_batch_shuffle = tf.gather(x_batch, unsupervised_mix_up_index)
        prob_shuffle = tf.gather(prob, unsupervised_mix_up_index)
        x_batch_mix = mix_weight * x_batch_shuffle + (1. - mix_weight) * x_batch
        pseudo_label = mix_weight * prob_shuffle + (1. - mix_weight) * prob
        smoothed_prob_mix = model.AE.get_prob(x_batch_mix)
        posterior_loss_y = - tf.reduce_mean(tf.reduce_sum(pseudo_label * tf.math.log(smoothed_prob_mix + eps), axis=-1))
        
        loss_unsupervised = ew * recon_loss + ucw * posterior_loss_y
        
    grad = tape.gradient(loss_unsupervised, model.AE.trainable_weights)
    optimizer[0].apply_gradients(zip(grad, model.AE.trainable_weights))
    if PARAMS['ema']:
        ema.apply(model.AE.trainable_weights)
    
    grad = tape.gradient(prior_loss, model.Prior.trainable_weights)
    optimizer[1].apply_gradients(zip(grad, model.Prior.trainable_weights))
    
    return [[loss_unsupervised, recon_loss, z_prior_loss, c_prior_loss, posterior_loss_y], 
            [z, c, xhat] + prior_args]
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

learning_rate_fn = [
    K.optimizers.schedules.PiecewiseConstantDecay(
    PARAMS['adjust_lr'], [PARAMS['learning_rate1'] * t for t in [1., 0.1, 0.01, 0.001]]
    ),
    K.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=PARAMS["learning_rate2"], 
        decay_steps=PARAMS['decay_steps'], 
        decay_rate=PARAMS['decay_rate']
    )
]

for epoch in range(PARAMS['epochs']):
    
    '''define optimizer and warm-up'''
    if epoch == 0:
        optimizer = [K.optimizers.SGD(learning_rate=PARAMS['learning_rate1'] * 0.2,
                                    momentum=PARAMS['beta_1']),
                     K.optimizers.Adam(learning_rate_fn[1](epoch), 
                                       clipvalue=PARAMS['gradclip'])]
    else:
        optimizer[0].lr.assign(learning_rate_fn[0](epoch))
        optimizer[1].lr.assign(learning_rate_fn[1](epoch))

    '''weights of loss terms'''
    # elbo part weight
    ew = weight_schedule(epoch, PARAMS['epochs'], PARAMS['ewm'])
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
                                                ew, mix_weight[0], optimizer) 
        
        '''mix-up: optimal match'''
        [[z, _, _, _], _] = model(x_batch)
        z = z.numpy()
        l2_metric = np.zeros((x_batch.shape[0], x_batch.shape[0]))
        for i in range(x_batch.shape[0]):
            l2_metric[i, :] = tf.reduce_sum(tf.square(z - z[i]), axis=1).numpy()
        unsupervised_mix_up_index = np.argsort(l2_metric, axis=1)[:, 1]
        
        '''unlabeled dataset training'''
        unsupervised_losses, unsupervised_outputs = unsupervised_train_step(x_batch, unsupervised_mix_up_index, PARAMS,
                                                                            ew, ucw, mix_weight[1], optimizer)
        
        step += 1
        
        progress_bar.set_description('epoch: {} | iteration {}/{} | supervised {:.3f}, unsupervised {:.3f}, z_prior {:.3f}, c_prior {:.3f}, test {:.3f}'.format(
            epoch, step, PARAMS['iterations'], 
            supervised_losses[0].numpy(), unsupervised_losses[0].numpy(), 
            supervised_losses[2].numpy() + unsupervised_losses[2].numpy(), 
            supervised_losses[3].numpy() + unsupervised_losses[3].numpy(), 
            test_error[-1]
        )) 
        
        if step % 10 == 0:
            print('epoch: {} | iteration {}/{} | supervised {:.3f}, unsupervised {:.3f}, z_prior {:.3f}, c_prior {:.3f}, test {:.3f}'.format(
                epoch, step, PARAMS['iterations'], 
                supervised_losses[0].numpy(), unsupervised_losses[0].numpy(), 
                supervised_losses[2].numpy() + unsupervised_losses[2].numpy(), 
                supervised_losses[3].numpy() + unsupervised_losses[3].numpy(), 
                test_error[-1]
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