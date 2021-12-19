#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class BasicBlock(K.layers.Layer):
    def __init__(self, 
                 in_filter_size, 
                 out_filter_size,
                 strides, 
                 slope=0.0,
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        
        self.slope = slope
        self.norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filters=out_filter_size, kernel_size=3, strides=strides, 
                                    padding='same', use_bias=False)
        self.norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=out_filter_size, kernel_size=3, strides=1, 
                                    padding='same', use_bias=False)
        
        self.equalInOut = (in_filter_size == out_filter_size)
        if not self.equalInOut:
            self.conv3 = layers.Conv2D(filters=out_filter_size, kernel_size=1, strides=strides, 
                                        padding='same', use_bias=False)

    @tf.function
    def call(self, x, training=True):
        if not self.equalInOut:
            x = tf.nn.leaky_relu(self.norm1(x, training=training), alpha=self.slope)
            h = tf.nn.leaky_relu(self.norm2(self.conv1(x), training=training), alpha=self.slope)
        else:
            h = tf.nn.leaky_relu(self.norm1(x, training=training), alpha=self.slope)
            h = tf.nn.leaky_relu(self.norm2(self.conv1(h), training=training), alpha=self.slope)
        h = self.conv2(h)
        if not self.equalInOut:
            h = h + self.conv3(x)
        else:
            h = h + x
        return h
#%%
class WideResNet(K.models.Model):
    def __init__(self, 
                 num_classes,
                 depth=28,
                 width=2,
                 slope=0.1,
                 input_shape=(None, 32, 32, 3),
                 name="WideResNet", 
                 **kwargs):
        super(WideResNet, self).__init__(input_shape, name=name, **kwargs)
        
        assert (depth - 4) % 6 == 0
        self.block_depth = (depth - 4) // 6
        self.nChannels = [16, 16*width, 32*width, 64*width]
        self.slope = slope
        
        # preprocess (small_input = True)
        self.conv = layers.Conv2D(filters=self.nChannels[0], kernel_size=3, strides=1, 
                                    padding='same', use_bias=False)
        
        # output: 32x32x32
        self.block1 = K.models.Sequential([BasicBlock(self.nChannels[0], self.nChannels[1], strides=1, slope=slope)] + \
                                            [BasicBlock(self.nChannels[1], self.nChannels[1], strides=1, slope=slope)])
        
        # output: 16x16x64
        self.block2 = K.models.Sequential([BasicBlock(self.nChannels[1], self.nChannels[2], strides=2, slope=slope)] + \
                                            [BasicBlock(self.nChannels[2], self.nChannels[2], strides=1, slope=slope)])
        
        # output: 8x8x128
        self.block3 = K.models.Sequential([BasicBlock(self.nChannels[2], self.nChannels[3], strides=2, slope=slope)] + \
                                            [BasicBlock(self.nChannels[3], self.nChannels[3], strides=1, slope=slope)])
        
        self.norm = layers.BatchNormalization()
        self.pooling = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(num_classes)
        
    def call(self, x, training=True):
        h = self.conv(x)
        h = self.block1(h, training=training)
        h = self.block2(h, training=training)
        h = self.block3(h, training=training)
        h = tf.nn.leaky_relu(self.norm(h, training=training), alpha=self.slope)
        h = self.pooling(h)
        h = self.dense(h)
        return h
#%%