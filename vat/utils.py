#%%
from multiprocessing import cpu_count

import tensorflow as tf
#%%
@tf.function
def augment(x, trans_range=2):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, paddings=[(0, 0),
                            (trans_range, trans_range),
                            (trans_range, trans_range), 
                            (0, 0)], mode='REFLECT')
    # x = tf.image.random_saturation(x, lower=0.6, upper=1.4)
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    return x
#%%
def kl_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit, axis=-1)
    logq = tf.nn.log_softmax(q_logit, axis=-1)
    logp = tf.nn.log_softmax(p_logit, axis=-1)
    
    qlogq = tf.reduce_sum(q * logq, axis=-1)
    qlogp = tf.reduce_sum(q * logp, axis=-1)
    kl = tf.reduce_mean(qlogq - qlogp)
    return kl
#%%
def _l2_normalize(d):
    d /= (tf.math.sqrt(tf.reduce_sum(tf.pow(d, 2.0), axis=[1, 2, 3], keepdims=True)) + 1e-6)
    return d
#%%
def generate_virtual_adversarial_perturbation(model, x, y, xi=1e-6, eps=8.0, num_iters=1):
    d = tf.random.normal(shape=(tf.shape(x))) # unit vector
    d = _l2_normalize(d)
    
    for i in range(num_iters):
        with tf.GradientTape() as tape:
            tape.watch(d)
            yhat = model(x + xi * d)
            dist = kl_with_logit(y, yhat)
        d = _l2_normalize(tape.gradient(dist, [d])[0])
    return eps * d
#%%