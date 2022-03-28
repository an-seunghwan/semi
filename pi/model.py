#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
#%%
class CNN(K.models.Model):
    def __init__(self, 
                 num_classes,
                 name="CNN", **kwargs):
        super(CNN, self).__init__(name=name, **kwargs)
        self.stddev = 0.15
        self.units = K.Sequential(
            [
                layers.Conv2D(filters=128, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=0.5),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=0.5),
                
                layers.Conv2D(filters=512, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, 
                                padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.GlobalAveragePooling2D(),
                
                layers.Dense(num_classes),
                layers.BatchNormalization(),
            ]
        )
    
    @tf.function
    def call(self, x, noise=True, training=True):
        if noise:
            noise = tf.random.normal(shape=tf.shape(x), stddev=self.stddev)
            x += noise
        h = self.units(x, training=training)
        h = tf.nn.softmax(h, axis=-1)
        return h
#%%
# model = VAT(10)
# model.build(input_shape=(None, 32, 32, 3))
# model.units.summary()
#%%