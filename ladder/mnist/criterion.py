#%%
import numpy as np
import tensorflow as tf
#%%
def ELBO_criterion(xhat, image, label, posterior, prior, num_classes, args):
    if args['bce_reconstruction']:
        error = - tf.reduce_sum(image * tf.math.log(tf.clip_by_value(xhat, 1e-10, 1.)) + 
                                (1. - image) * tf.math.log(1. - tf.clip_by_value(xhat, 1e-10, 1.)), axis=[1, 2, 3])
    else:
        error = tf.reduce_sum(tf.math.square(image - xhat), axis=[1, 2, 3]) / 2.
    
    prior_y = tf.reduce_sum(label * tf.math.log(1 / num_classes), axis=-1)
    
    kl = 0.
    for (z, qmean, qlogvar), (pmean, plogvar) in zip(posterior, prior):
        pz = tf.reduce_sum(- 0.5 * tf.math.log(2 * np.math.pi) - 0.5 * plogvar  - 0.5 * ((z - pmean) ** 2) / tf.math.exp(plogvar), axis=-1)
        qz = tf.reduce_sum(- 0.5 * tf.math.log(2 * np.math.pi) - 0.5 * qlogvar  - 0.5 * ((z - qmean) ** 2) / tf.math.exp(qlogvar), axis=-1)
        kl += qz - pz
    
    return error, prior_y, kl
#%%