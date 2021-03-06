#%%
from multiprocessing import cpu_count

import tensorflow as tf
#%%
@tf.function
def augment(x, trans_range):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, paddings=[(0, 0),
                            (trans_range, trans_range),
                            (trans_range, trans_range), 
                            (0, 0)], mode='REFLECT')
    # x = tf.image.random_saturation(x, lower=0.6, upper=1.4)
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    return x
#%%