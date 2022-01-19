#%%
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
#%%
@tf.function
def augment(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, paddings=[(0, 0),
                            (4, 4),
                            (4, 4), 
                            (0, 0)], mode='REFLECT')
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    return x
#%%
def label_smoothing(image, z, c, label, mix_weight):
    shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
    image_shuffle = tf.gather(image, shuffled_indices)
    z_shuffle = tf.gather(z, shuffled_indices)
    c_shuffle = tf.gather(c, shuffled_indices)
    label_mix = tf.gather(label, shuffled_indices)
    
    image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
    z_mix = mix_weight * z_shuffle + (1. - mix_weight) * z
    c_mix = mix_weight * c_shuffle + (1. - mix_weight) * c
    
    return image_mix, z_mix, c_mix, label_mix
#%%
def non_smooth_mixup(image, z, c, prob, mix_weight):
    shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
    image_shuffle = tf.gather(image, shuffled_indices)
    z_shuffle = tf.gather(z, shuffled_indices)
    c_shuffle = tf.gather(c, shuffled_indices)
    prob_shuffle = tf.gather(prob, shuffled_indices)
    
    image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
    z_mix = mix_weight * z_shuffle + (1. - mix_weight) * z
    c_mix = mix_weight * c_shuffle + (1. - mix_weight) * c
    prob_mix = mix_weight * prob_shuffle + (1. - mix_weight) * prob
    
    return image_mix, z_mix, c_mix, prob_mix
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        buffer_var.assign(var)
        
def weight_decay(model, decay_rate):
    for var in model.trainable_variables:
        var.assign(var * (1. - decay_rate))
#%%