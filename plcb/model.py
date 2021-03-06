#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
#%%
class CNN(K.models.Model):
    def __init__(self, 
                 num_classes=10,
                 dropratio=0.0,
                 isL2=False,
                 name="CNN", **kwargs):
        super(CNN, self).__init__(name=name, **kwargs)
        self.isL2 = isL2
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
                layers.SpatialDropout2D(rate=dropratio),
                
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
                layers.SpatialDropout2D(rate=dropratio),
                
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
            ]
        )
        self.fc = layers.Dense(num_classes)
    
    @tf.function
    def call(self, x, training=True):
        h = self.units(x, training=training)
        if self.isL2:
            h /= tf.math.maximum(tf.norm(h, axis=-1, keepdims=True), 1e-12)
        h = self.fc(h)
        return h
#%%
# model = CNN(10)
# model.build(input_shape=(None, 32, 32, 3))
# model.units.summary()
#%%