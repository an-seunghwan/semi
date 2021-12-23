#%%
from multiprocessing import cpu_count

# import numpy as np
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
# def gaussian_kl_divergence(mean1, mean2, log_sigma1, log_sigma2):
#     return 0.5 * tf.reduce_sum(tf.math.square(mean1 - mean2) / tf.math.exp(2. * log_sigma2)
#                                + tf.math.exp(2. * log_sigma1) / tf.math.exp(2. * log_sigma2) 
#                                - 1 
#                                + 2. * log_sigma2 
#                                - 2. * log_sigma1, axis=-1)
#%%
# def label_smoothing(image, label, mean, log_sigma, mix_weight):
#     # no-gradient error!!!
#     # x_batch_L_shuffle = tf.random.shuffle(x_batch_L)
#     # y_batch_L_shuffle = tf.random.shuffle(y_batch_L)
#     # mean_shuffle = tf.random.shuffle(mean)
#     # logvar_shuffle = tf.random.shuffle(logvar)
#     image_shuffle = tf.gather(image, tf.random.shuffle(tf.range(tf.shape(image)[0])))
#     label_shuffle = tf.gather(label, tf.random.shuffle(tf.range(tf.shape(label)[0])))
#     mean_shuffle = tf.gather(mean, tf.random.shuffle(tf.range(tf.shape(mean)[0])))
#     log_sigma_shuffle = tf.gather(log_sigma, tf.random.shuffle(tf.range(tf.shape(log_sigma)[0])))
    
#     image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
#     mean_mix = mix_weight * mean_shuffle + (1. - mix_weight) * mean
#     sigma_mix = mix_weight * tf.math.exp(log_sigma_shuffle) + (1. - mix_weight) * tf.math.exp(log_sigma)
    
#     return image_mix, label_shuffle, mean_mix, sigma_mix
#%%
# def optimal_match_mix(image, mean, log_sigma, log_prob, mix_weight):
#     kl_metric = np.zeros((tf.shape(mean)[0], tf.shape(mean)[0]))
#     for i in range(tf.shape(mean)[0]):
#         kl_metric[i, :] = gaussian_kl_divergence(mean, mean.numpy()[i], log_sigma, log_sigma.numpy()[i]).numpy()
        
#     unsupervised_mix_up_index = np.argsort(kl_metric, axis=1)[:, 1]
#     image_shuffle = tf.gather(image, unsupervised_mix_up_index)
#     mean_shuffle = tf.gather(mean, unsupervised_mix_up_index)
#     log_sigma_shuffle = tf.gather(log_sigma, unsupervised_mix_up_index)
#     log_prob_shuffle = tf.gather(log_prob, unsupervised_mix_up_index)
    
#     image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
#     mean_mix = mix_weight * mean_shuffle + (1. - mix_weight) * mean
#     sigma_mix = mix_weight * tf.math.exp(log_sigma_shuffle) + (1. - mix_weight) * tf.math.exp(log_sigma)
#     pseudo_label = mix_weight * tf.math.exp(log_prob_shuffle) + (1. - mix_weight) * tf.math.exp(log_prob)
    
#     return image_mix, mean_mix, sigma_mix, pseudo_label
#%%
def weight_decay(model, decay_model, decay_rate):
    for var, decay_var in zip(model.variables, decay_model.variables):
        decay_var.assign(var - decay_rate * decay_var)
            
# def weight_decay(model, decay_rate):
#     for var in model.trainable_variables:
#         var.assign(var * (1 - decay_rate))
#%%