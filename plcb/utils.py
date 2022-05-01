#%%
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
#%%
@tf.function
def augment(x, trans_range=2):
    x = tf.pad(x, paddings=[(0, 0),
                            (trans_range, trans_range),
                            (trans_range, trans_range), 
                            (0, 0)], mode='REFLECT')
    # color jitter
    x = tf.image.random_hue(x, 0.1)
    x = tf.image.random_saturation(x, 0.6, 1.4)
    x = tf.image.random_brightness(x, 0.4)
    x = tf.image.random_contrast(x, 0.6, 1.4)
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    x = tf.image.random_flip_left_right(x)
    return x
#%%
def non_smooth_mixup(image, label, mix_weight):
    shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
    image_shuffle = tf.gather(image, shuffled_indices)
    label_shuffle = tf.gather(label, shuffled_indices)
    
    image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
    
    return image_mix, label_shuffle
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        buffer_var.assign(var)
#%%