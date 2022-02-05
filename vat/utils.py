#%%
import tensorflow as tf
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
def generate_virtual_adversarial_perturbation(model, x, y, xi=1e-6, eps=2.5, num_iters=1):
    d = _l2_normalize(tf.random.normal(shape=(tf.shape(x)))) # unit vector
    for i in range(num_iters):
        with tf.GradientTape() as tape:
            r = xi * d 
            tape.watch(r)
            yhat = model(x + r, training=False)
            dist = kl_with_logit(y, yhat)
        grad = tape.gradient(dist, [r])[0]
        d = _l2_normalize(tf.stop_gradient(grad))
    return eps * d
#%%