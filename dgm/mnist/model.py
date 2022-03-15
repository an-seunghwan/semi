#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class Encoder(K.models.Model):
    def __init__(self, latent_dim , name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.net = K.Sequential(
            [
                layers.Dense(256, activation='linear'),
                layers.ReLU(),
                layers.Dense(128, activation='linear'),
                layers.ReLU(),
            ]
        )
        
        self.mean_layer = layers.Dense(latent_dim, activation='linear')
        self.logvar_layer = layers.Dense(latent_dim, activation='softplus')
    
    # @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar
#%%
# class Classifier(K.models.Model):
#     def __init__(self, num_classes, name="Classifier", **kwargs):
#         super(Classifier, self).__init__(name=name, **kwargs)
#         self.net = K.Sequential(
#             [
#                 layers.Flatten(),
#                 layers.Dense(256, activation='linear'),
#                 layers.ReLU(),
#                 layers.Dense(num_classes, activation='softmax'),
#             ]
#         )
        
#     # @tf.function
#     def call(self, x, training=True):
#         h = self.net(x, training=training)
#         return h
#%%
class Classifier(K.models.Model):
    def __init__(self, num_classes, name="Classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.nets = K.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same'), 
                layers.BatchNormalization(),
                layers.LeakyReLU(0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=0.5),
                
                layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'), 
                layers.BatchNormalization(),
                layers.LeakyReLU(0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=0.5),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'), 
                layers.BatchNormalization(),
                layers.LeakyReLU(0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=0.5),
                
                layers.GlobalAveragePooling2D(),
                
                layers.Dense(64, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dense(num_classes, activation='softmax'),
            ]
        )
    
    # @tf.function
    def call(self, x, training=True):
        h = self.nets(x, training=training)
        return h
#%%
class Decoder(K.models.Model):
    def __init__(self, activation='sigmoid', name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.net = K.Sequential(
            [
                layers.Dense(128, activation='linear'),
                layers.ReLU(),
                layers.Dense(256, activation='linear'),
                layers.ReLU(),
                layers.Dense(784, activation=activation),
                layers.Reshape((28, 28, 1)),
            ]
        )
    
    # @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        return h
#%%
class DGM(K.models.Model):
    def __init__(self, 
                 args,
                 num_classes=10,
                 latent_dim=2, 
                 activation='sigmoid',
                 input_dim=(None, 28, 28, 1), 
                 name='DGM', **kwargs):
        super(DGM, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.encoder = Encoder(latent_dim)
        self.classifier = Classifier(num_classes)
        self.decoder = Decoder(activation)
    
    def encode(self, inputs, training=True):
        x, y = inputs
        x = layers.Flatten()(x)
        h = tf.concat([x, y], axis=-1)
        mean, logvar = self.encoder(h, training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        return mean, logvar, z
    
    def classify(self, x, training=True):
        prob = self.classifier(x, training=training)
        return prob
    
    def decode(self, z, y, training=True):
        xhat = self.decoder(tf.concat([z, y], axis=-1), training=training) 
        return xhat
        
    @tf.function
    def call(self, inputs, training=True):
        x, y = inputs
        x = layers.Flatten()(x)
        h = tf.concat([x, y], axis=-1)
        mean, logvar = self.encoder(h, training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        # assert z.shape == (tf.shape(x)[0], self.latent_dim)
        xhat = self.decoder(tf.concat([z, y], axis=-1), training=training) 
        # assert xhat.shape == (tf.shape(x)[0], self.input_dim[1], self.input_dim[2], self.input_dim[3])
        return mean, logvar, z, xhat
#%%