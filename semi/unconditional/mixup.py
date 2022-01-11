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
def label_smoothing(image, label, mix_weight, tag):
    shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
    image_shuffle = tf.gather(image, shuffled_indices)
    
    image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
    if tag == 'labeled':
        label_mix = tf.gather(label, shuffled_indices)
    if tag == 'unlabeled':
        label_shuffle = tf.gather(label, shuffled_indices)
        label_mix = mix_weight * label_shuffle + (1. - mix_weight) * label
    
    return image_mix, label_mix
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.variables, buffer_model.variables):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.variables, buffer_model.variables):
        buffer_var.assign(var)
        
def weight_decay(model, decay_rate):
    for var in model.trainable_variables:
        var.assign(var * (1. - decay_rate))
#%%