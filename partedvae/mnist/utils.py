#%%
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

import random
import scipy
#%%
# @tf.function
# def augment(x):
#     x = tf.image.random_flip_left_right(x)
#     x = tf.pad(x, paddings=[(0, 0),
#                             (4, 4),
#                             (4, 4), 
#                             (0, 0)], mode='REFLECT')
#     x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
#     return x
#%%
# Crops the center of the image
def crop_center(img, cropx, cropy):
    if len(img.shape) == 2:
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]
    elif len(img.shape) == 3:
        y, x, b = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx, :]

# Take a random crop of the image
def crop_random(img, cropx, cropy):
    # takes numpy input
    if len(img.shape) == 2:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1 : x1 + cropx, y1 : y1 + cropy]
    elif len(img.shape) == 3:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1 : x1 + cropx, y1 : y1 + cropy, :]

def augment(image):
    npimage = image.numpy() # input is tensor
    rotation = [random.randrange(-25, 25) for i in range(len(npimage))]
    rotatedImg = [scipy.ndimage.interpolation.rotate(im, rotation[i], axes=(0, 1), mode='nearest') for i, im in enumerate(npimage)]
    # crop image to 28x28 as rotation increases size
    rotatedImgCentered = [crop_center(im, 28, 28) for im in rotatedImg]
    # pad image by 3 pixels on each edge (-0.42421296 background color)
    paddedImg = [np.pad(im, ((4, 4), (4, 4), (0, 0)), 'constant', constant_values=-1) for im in rotatedImgCentered]
    # randomly crop from padded image
    cropped = np.array([crop_random(im, 28, 28) for im in paddedImg])
    return tf.stack(cropped, axis=0)
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        buffer_var.assign(var)
        
# def weight_decay(model, decay_rate):
#     for var in model.trainable_variables:
#         var.assign(var * (1. - decay_rate))
#%%
# Original Callback found in tf.keras.callbacks.Callback
# Copyright The TensorFlow Authors and Keras Authors.

class CustomReduceLRoP():

    """ Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This callback monitors a
        quantity and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.
        Example:
        ```python
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                        patience=5, min_lr=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```
    Arguments:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced. new_lr = lr *
            factor
        patience: number of epochs with no improvement after which learning rate
            will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
            quantity monitored has stopped decreasing; in `max` mode it will be
            reduced when the quantity monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred from the name of the
            monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
            significant changes.
        cooldown: number of epochs to wait before resuming normal operation after
            lr has been reduced.
        min_lr: lower bound on the learning rate.
        reduce_exp: reducing the learning rate exponentially
    """

    def __init__(self,
                 ## Custom modification:  Deprecated due to focusing on validation loss
                 # monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 sign_number = 4,
                 ## Custom modification: Passing optimizer as arguement
                 optim_lr = None,
                 ## Custom modification:  Exponentially reducing learning
                 reduce_lin = False,
                 **kwargs):

        ## Custom modification:  Deprecated
        # super(ReduceLROnPlateau, self).__init__()

        ## Custom modification:  Deprecated
        # self.monitor = monitor
        
        ## Custom modification: Optimizer Error Handling
        if tf.is_tensor(optim_lr) == False:
            raise ValueError('Need optimizer !')
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        ## Custom modification: Passing optimizer as arguement
        self.optim_lr = optim_lr  

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.sign_number = sign_number
        

        ## Custom modification: Exponentially reducing learning
        self.reduce_lin = reduce_lin
        self.reduce_lr = True
        

        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            print('Learning Rate Plateau Reducing mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                ## Custom modification: Deprecated due to focusing on validation loss
                # (self.mode == 'auto' and 'acc' not in self.monitor)):
                (self.mode == 'auto')):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, loss, logs=None):


        logs = logs or {}
        ## Custom modification: Optimizer
        # logs['lr'] = K.get_value(self.model.optimizer.lr) returns a numpy array
        # and therefore can be modified to          
        logs['lr'] = float(self.optim_lr.numpy())

        ## Custom modification: Deprecated due to focusing on validation loss
        # current = logs.get(self.monitor)

        current = float(loss)
        
        ## Custom modification: Deprecated due to focusing on validation loss
        # if current is None:
        #     print('Reduce LR on plateau conditioned on metric `%s` '
        #                     'which is not available. Available metrics are: %s',
        #                     self.monitor, ','.join(list(logs.keys())))

        # else:

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                
                ## Custom modification: Optimizer Learning Rate
                # old_lr = float(K.get_value(self.model.optimizer.lr))
                old_lr = float(self.optim_lr.numpy())
                if old_lr > self.min_lr and self.reduce_lr == True:
                    ## Custom modification: Linear learning Rate
                    if self.reduce_lin == True:
                        new_lr = old_lr - self.factor
                        ## Custom modification: Error Handling when learning rate is below zero
                        if new_lr <= 0:
                            print('Learning Rate is below zero: {}, '
                            'fallback to minimal learning rate: {}. '
                            'Stop reducing learning rate during training.'.format(new_lr, self.min_lr))  
                            self.reduce_lr = False                           
                    else:
                        new_lr = old_lr * self.factor                   
                    

                    new_lr = max(new_lr, self.min_lr)


                    ## Custom modification: Optimizer Learning Rate
                    # K.set_value(self.model.optimizer.lr, new_lr)
                    self.optim_lr.assign(new_lr)
                    
                    if self.verbose > 0:
                        print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                'rate to %s.' % (epoch + 1, float(new_lr)))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
#%%