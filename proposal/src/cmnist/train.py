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
import datetime
from pprint import pprint
from itertools import cycle
import json
import io
import cv2
import os
os.chdir(r'D:\semi\proposal')
# os.chdir('/home1/prof/jeon/an/semi/proposal')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from modules import CMNIST
#%%
PARAMS = {
    "data": 'cmnist',
    "batch_size": 128,
    "epochs": 600,
    "data_dim": 32,
    "channel": 3, 
    "class_num": 10,
    "annotated_ratio": 0.1,
    
    # "ewm": 1e-3, # elbo weight max
    # "aew": 400, # the epoch to adjust elbo weight to max
    # "pwm": 1, # posterior weight max
    # "apw": 200, # adjust posterior weight
    # "wrd": 1., # the max weight for the optimal transport estimation of discrete variable 
    # "wmf": 0.4, # the weight factor: epoch to adjust the weight for the optimal transport estimation of discrete variable to max
    
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
    "decay_steps": 20,
    "decay_rate": 0.95,
    "gradclip": 1.,
    "BN_in_NF": False,
    
    "beta": 1.,
    "lambda1": 10., 
    "lambda2": 10., 
    "learning_rate1": 0.001, 
    "learning_rate2": 0.0001,
    "beta_1": 0.9, # beta_1 in SGD or Adam
    # "adjust_lr": [400, 500, 550], # the milestone list for adjust learning rate
    "weight_decay": 5e-4, 
    "epsilon": 0.1, # the label smoothing epsilon for labeled dataset
    "activation": 'sigmoid',
    "observation": 'mse',
    "ema": True,
    
    "slope": 0.01, # pytorch default
    "widen_factor": 2,
    "depth": 28,
    "decoder_feature": 8,
}
PARAMS['z_nf_dim'] = PARAMS['z_dim'] // 2
PARAMS['c_nf_dim'] = PARAMS['c_dim'] // 2
#%%
model = CMNIST.DeterministicVAE(PARAMS) 
model.Prior.build_graph()
#%%
'''dataset'''
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
#%%
'''Colored & Edged MNIST'''
def colored_mnist(image, PARAMS):
    if tf.random.uniform((1, 1)) > 0.5:
        '''color'''
        image = tf.cast(image, tf.float32) / 255.
        color = np.random.uniform(0., 1., 3)
        color = tf.cast(color / tf.linalg.norm(color), tf.float32)
        image = image[..., tf.newaxis] * color[tf.newaxis, tf.newaxis, :]
        
        '''resize'''
        image = tf.image.resize(image, [PARAMS['data_dim'], PARAMS['data_dim']], method='bilinear')
        
        assert image.shape == (PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel'])
        if PARAMS['activation'] == 'tanh':
            return image * 2. - 1.
        elif PARAMS['activation'] == 'sigmoid':
            return image
        else:
            assert 0, "Unsupported observation model: {}".format(PARAMS['observation'])
    else:
        '''edge detection'''
        image = cv2.Canny(image, 10., 255.)
        image[np.where(image > 0)] = 1.
        image[np.where(image <= 0)] = 0.
        
        '''color'''
        color = np.random.uniform(0., 1., 3)
        color = color / np.linalg.norm(color)
        image = image[..., tf.newaxis] * color[tf.newaxis, tf.newaxis, :]
        
        '''width'''
        kernel = np.ones((1, 1))
        image = cv2.dilate(image, kernel)
        
        '''resize'''
        image = tf.image.resize(image, [PARAMS['data_dim'], PARAMS['data_dim']], method='bilinear')

        assert image.shape == (PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel'])
        if PARAMS['activation'] == 'tanh':
            return image * 2. - 1.
        elif PARAMS['activation'] == 'sigmoid':
            return image
        else:
            assert 0, "Unsupported observation model: {}".format(PARAMS['observation'])
#%%
tf.random.set_seed(1)
cx_train = []
for i in tqdm(range(len(x_train)), desc='generating train colored mnist'):
    cx_train.append(colored_mnist(x_train[i], PARAMS))
cx_train = np.array(cx_train)

# plt.imshow(cx_train[10])
# plt.show()

cx_test = []
for i in tqdm(range(len(x_test)), desc='generating test colored mnist'):
    cx_test.append(colored_mnist(x_test[i], PARAMS))
cx_test = np.array(cx_test)
#%%
x_train = cx_train.astype('float32')
x_test = cx_test.astype('float32') 
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

del cx_train
del cx_test
#%%
# def weight_schedule(epoch, epochs, weight_max):
#     return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
'''Define our metrics'''
train_loss = K.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = K.metrics.CategoricalAccuracy('train_accuracy')
train_recon = K.metrics.Mean('train_recon', dtype=tf.float32)
train_z_prior = K.metrics.Mean('train_z_prior', dtype=tf.float32)
train_c_prior = K.metrics.Mean('train_c_prior', dtype=tf.float32)
train_prior = K.metrics.Mean('train_prior', dtype=tf.float32)
test_accuracy = K.metrics.CategoricalAccuracy('test_accuracy')
#%%
log_dir = 'logs/{}/{}'.format(PARAMS['data'], current_time)
summary_writer = tf.summary.create_file_writer(log_dir)
asset_path = 'weights_{}'.format(current_time)
pprint(PARAMS)
#%%
# @tf.function
def supervised_loss(outputs, x_batch_L, y_batch_L, 
                    mix_weight,
                    PARAMS):
    eps = 1e-8
    
    [[z, c, prob, xhat], prior_args] = outputs
    
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
    
    loss_supervised = recon_loss / PARAMS['beta'] + PARAMS['lambda1'] * classification_loss + posterior_loss_y
    
    return [loss_supervised, recon_loss, z_prior_loss, c_prior_loss, classification_loss, prior_loss]
#%%
# @tf.function
def supervised_train_step(x_batch_L, y_batch_L, PARAMS,
                          mix_weight,
                          optimizer, optimizer_NF):
    
    with tf.GradientTape(persistent=True) as tape:
        supervised_outputs = model(x_batch_L)
        supervised_losses = supervised_loss(supervised_outputs, x_batch_L, y_batch_L, mix_weight, PARAMS)
        
    grad = tape.gradient(supervised_losses[0], model.AE.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.AE.trainable_weights))
    grad = tape.gradient(supervised_losses[-1], model.Prior.trainable_weights)
    optimizer_NF.apply_gradients(zip(grad, model.Prior.trainable_weights))
    
    train_accuracy(y_batch_L, supervised_outputs[0][2])
    
    return supervised_losses, supervised_outputs
#%%
# @tf.function
def unsupervised_loss(outputs, x_batch, 
                      unsupervised_mix_up_index, mix_weight,
                      PARAMS):
    eps = 1e-8
    
    [[z, c, prob, xhat], prior_args] = outputs
    
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
    
    loss_unsupervised = recon_loss / PARAMS['beta'] + posterior_loss_y
    
    return [loss_unsupervised, recon_loss, z_prior_loss, c_prior_loss, prior_loss]
#%%
# @tf.function
def unsupervised_train_step(x_batch, PARAMS,
                            unsupervised_mix_up_index, mix_weight, 
                            optimizer, optimizer_NF):
    
    with tf.GradientTape(persistent=True) as tape:
        unsupervised_outputs = model(x_batch)
        unsupervised_losses = unsupervised_loss(unsupervised_outputs, x_batch, unsupervised_mix_up_index, mix_weight, PARAMS)
        
    grad = tape.gradient(unsupervised_losses[0], model.AE.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.AE.trainable_weights))
    grad = tape.gradient(unsupervised_losses[-1], model.Prior.trainable_weights)
    optimizer_NF.apply_gradients(zip(grad, model.Prior.trainable_weights))
    
    train_loss(unsupervised_losses[0] + unsupervised_losses[-1])
    train_recon(unsupervised_losses[1])
    train_z_prior(unsupervised_losses[2])
    train_c_prior(unsupervised_losses[3])
    train_prior(unsupervised_losses[-1])
    
    return unsupervised_losses, unsupervised_outputs
#%%
def test_step(model, x_test_batch, y_test_batch):
    '''test classification error''' 
    predictions = model.AE.get_prob(x_test_batch, training=False)
    test_accuracy(y_test_batch, predictions)
#%%
def generate_and_save_images(x_batch):
    x = x_batch[0][tf.newaxis, ...]
    z = model.AE.get_latent(x, training=False)
    
    buf = io.BytesIO()
    figure = plt.figure(figsize=(10, 2))
    plt.subplot(1, PARAMS['class_num']+1, 1)
    plt.imshow((x[0] + 1) / 2)
    plt.title('original')
    plt.axis('off')
    for i in range(PARAMS['class_num']):
        label = np.zeros((z.shape[0], PARAMS['class_num']))
        label[:, i] = 1
        xhat = model.AE.Decoder(z, label, training=False)
        plt.subplot(1, PARAMS['class_num']+1, i+2)
        plt.imshow((xhat[0] + 1) / 2)
        plt.title('{}'.format(i))
        plt.axis('off')
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
#%%
learning_rate_fn = K.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=PARAMS["learning_rate2"], 
    decay_steps=PARAMS['decay_steps'], 
    decay_rate=PARAMS['decay_rate']
)

for epoch in range(PARAMS['epochs']):
    
    '''define optimizer and warm-up'''
    if epoch == 0:
        optimizer = K.optimizers.Adam(learning_rate=PARAMS['learning_rate1'] * 0.2,
                                    beta_1=PARAMS['beta_1'])
        optimizer_NF = K.optimizers.Adam(learning_rate_fn(epoch), 
                                       clipvalue=PARAMS['gradclip'])
    else:
        optimizer.lr = PARAMS['learning_rate1']
        optimizer_NF.lr = learning_rate_fn(epoch)

    # '''weights of loss terms'''
    # # elbo part weight
    # ew = weight_schedule(epoch, PARAMS['aew'], PARAMS['ewm'])
    # # optimal transport weight
    # ucw = weight_schedule(epoch, round(PARAMS['wmf'] * PARAMS['epochs']), PARAMS['wrd'])
    
    for (x_batch_L, y_batch_L), x_batch in tqdm(zip(cycle(train_L_dataset), train_dataset), total=len(train_dataset)):
    
        '''mix-up weight'''
        mix_weight = [tf.constant(np.random.beta(PARAMS['epsilon'], PARAMS['epsilon'])), # labeled
                      tf.constant(np.random.beta(2.0, 2.0))] # unlabeled
        
        '''labeled dataset training'''
        supervised_losses, supervised_outputs = supervised_train_step(x_batch_L, y_batch_L, PARAMS,
                                                mix_weight[0], 
                                                optimizer, optimizer_NF) 
        
        '''mix-up: optimal match'''
        [[z, _, _, _], _] = model(x_batch)
        z = z.numpy()
        l2_metric = np.zeros((x_batch.shape[0], x_batch.shape[0]))
        for i in range(x_batch.shape[0]):
            l2_metric[i, :] = tf.reduce_sum(tf.square(z - z[i]), axis=1).numpy()
        unsupervised_mix_up_index = np.argsort(l2_metric, axis=1)[:, 1]
        
        '''unlabeled dataset training'''
        unsupervised_losses, unsupervised_outputs = unsupervised_train_step(x_batch, PARAMS,
                                                                            unsupervised_mix_up_index, mix_weight[1], 
                                                                            optimizer, optimizer_NF)
    with summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
        tf.summary.scalar('reconstruction', train_recon.result(), step=epoch)
        tf.summary.scalar('z_prior', train_z_prior.result(), step=epoch)
        tf.summary.scalar('c_prior', train_c_prior.result(), step=epoch)
        tf.summary.scalar('prior', train_prior.result(), step=epoch)
        if (epoch + 1) % 50 == 0:
            tf.summary.image("train recon image", generate_and_save_images(x_batch), step=epoch)   
        
    for (x_test_batch, y_test_batch) in test_dataset:
        test_step(model, x_test_batch, y_test_batch)
    with summary_writer.as_default():
        tf.summary.scalar('test_accuracy', test_accuracy.result(), step=epoch)
        
    template = 'Epoch {}, loss {:.3f}, recon {:.3f}, z_prior {:.3f}, c_prior {:.3f}, train accuracy {:.3f}%. test accuracy {:.3f}%'
    print(template.format(epoch+1,
                        train_loss.result(), 
                        train_recon.result(),
                        train_z_prior.result(),
                        train_c_prior.result(),
                        train_accuracy.result()*100,
                        test_accuracy.result()*100))

    # Reset metrics every epoch
    train_loss.reset_states()
    train_recon.reset_states()
    train_z_prior.reset_states()
    train_c_prior.reset_states()
    train_prior.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
#%%
'''save model and configs'''
# model.save_weights('./assets/{}/{}/weights'.format(PARAMS['data'], asset_path))
os.makedirs('./assets/{}/{}'.format(PARAMS['data'], asset_path))
model.save_weights('./assets/{}/{}/model.h5'.format(PARAMS['data'], asset_path), save_format="h5")
model.summary()

with open('./assets/{}/{}/params.json'.format(PARAMS['data'], asset_path), 'w') as f:
    json.dump(PARAMS, f, indent=4, separators=(',', ': '))
#%%
'''import model'''
# imported = CIFAR10.VAE(PARAMS)
# dummy = tf.random.normal((1, 32, 32, 3))
# '''call the model first'''
# _ = imported(dummy)
# imported.load_weights('./assets/{}/{}/model.h5'.format(PARAMS['data'], asset_path))
# imported(x_batch_L)
#%%