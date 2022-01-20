#%%
import tensorflow as tf
#%%
def ELBO_criterion(args, x, xhat, prob, prob_recon, label=None):
        '''reconstruction'''
        if args['br']:
            recon_loss = tf.reduce_mean(- tf.reduce_sum(x * tf.math.log(tf.clip_by_value(xhat, 1e-10, 1.0)) + 
                                                        (1. - x) * tf.math.log(tf.clip_by_value(1. - xhat, 1e-10, 1.0)), axis=[1, 2, 3]))
        else:
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(xhat - x) / (2. * (args['x_sigma'] ** 2)), axis=[1, 2, 3]))
        
        if label is not None:
            '''classification'''
            cls_loss = tf.reduce_mean(- tf.reduce_sum(label * tf.math.log(tf.clip_by_value(prob, 1e-10, 1.0)), axis=-1))
            '''mutual information'''
            info = tf.reduce_mean(- tf.reduce_sum(label * tf.math.log(tf.clip_by_value(prob_recon, 1e-10, 1.0)), axis=-1))
        else:
            cls_loss = 0
            '''mutual information'''
            info = tf.reduce_mean(- tf.reduce_sum(prob * tf.math.log(tf.clip_by_value(prob_recon, 1e-10, 1.0)), axis=-1))
        
        # '''prior'''
        # z_nf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(nf_args[0] - 0) / 2., axis=1))
        # z_nf_loss -= tf.reduce_mean(nf_args[1], axis=-1)
        # c_nf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(nf_args[2] - 0) / 2., axis=1))
        # c_nf_loss -= tf.reduce_mean(nf_args[3], axis=-1)
        # nf_loss = z_nf_loss + c_nf_loss
        
        # return recon_loss, cls_loss, info, nf_loss
        return recon_loss, cls_loss, info,
#%%