#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class ResidualUnit(K.layers.Layer):
    def __init__(self, 
                 filter_in, 
                 filter_out,
                 strides, 
                 slope=0.1,
                 **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        
        self.norm1 = layers.BatchNormalization()
        self.relu1 = layers.LeakyReLU(alpha=slope)
        self.conv1 = layers.Conv2D(filters=filter_out, kernel_size=3, strides=strides, 
                                    padding='same', use_bias=False)
        self.norm2 = layers.BatchNormalization()
        self.relu2 = layers.LeakyReLU(alpha=slope)
        self.conv2 = layers.Conv2D(filters=filter_in, kernel_size=3, strides=1, 
                                    padding='same', use_bias=False)
        
        self.equalInOut = (filter_in == filter_out)
        if not self.equalInOut:
            self.shortcut = layers.Conv2D(filters=filter_out, kernel_size=1, strides=strides, 
                                        padding='same', use_bias=False)

    @tf.function
    def call(self, x, training=True):
        if not self.equalInOut:
            x = self.relu1(self.norm1(x, training=training))
            h = self.relu2(self.norm2(self.conv1(x), training=training))
        else:
            h = self.relu1(self.norm1(x, training=training))
            h = self.relu2(self.norm2(self.conv1(h), training=training))
        h = self.conv2(h)
        if not self.equalInOut:
            h = h + self.shortcut(x)
        else:
            h = h + x
        return h
#%%
class ResidualBlock(K.layers.Layer):
    def __init__(self,
                 n_units,
                 filter_in,
                 filter_out,
                 unit,
                 strides, 
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = self._build_unit(n_units, unit, filter_in, filter_out, strides)
    
    def _build_unit(self, n_units, unit, filter_in, filter_out, strides):
        units = []
        for i in range(n_units):
            units.append(unit(filter_in if i == 0 else filter_out, filter_out, strides if i == 0 else 1))
        return units
    
    @tf.function
    def call(self, x, training=True):
        for unit in self.units:
            x = unit(x, training=training)
        return x
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
        self.n_units = (depth - 4) // 6
        self.nChannels = [16, 16*width, 32*width, 64*width]
        self.slope = slope
        
        # preprocess (small_input = True)
        self.conv = layers.Conv2D(filters=self.nChannels[0], kernel_size=3, strides=1, 
                                    padding='same', use_bias=False)
        
        self.block1 = ResidualBlock(self.n_units, self.nChannels[0], self.nChannels[1], ResidualUnit, 1)
        self.block2 = ResidualBlock(self.n_units, self.nChannels[1], self.nChannels[2], ResidualUnit, 2)
        self.block3 = ResidualBlock(self.n_units, self.nChannels[2], self.nChannels[3], ResidualUnit, 2)
        
        self.norm = layers.BatchNormalization()
        self.relu = layers.LeakyReLU(alpha=slope)
        self.pooling = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(num_classes)
        
    def call(self, x, training=True):
        h = self.conv(x)
        h = self.block1(h, training=training)
        h = self.block2(h, training=training)
        h = self.block3(h, training=training)
        h = self.relu(self.norm(h, training=training))
        h = self.pooling(h)
        h = self.dense(h)
        return h
#%%