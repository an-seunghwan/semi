#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class auxEncoder(K.models.Model):
    def __init__(self, a_dim, **kwargs):
        super(auxEncoder, self).__init__(**kwargs)
        self.net = K.Sequential(
            [
                layers.Dense(256, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dense(128, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.mean_layer = layers.Dense(a_dim, activation='linear')
        self.logvar_layer = layers.Dense(a_dim, activation='softplus')
    
    # @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar
#%%
class Classifier(K.models.Model):
    def __init__(self, num_classes, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.dense_x = layers.Dense(256, activation='linear')
        self.dense_a = layers.Dense(256, activation='linear')
        self.net = K.Sequential(
            [   
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dropout(rate=0.5),
                layers.Dense(128, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dropout(rate=0.5),
                layers.Dense(num_classes, activation='softmax'),
            ]
        )
        
    # @tf.function
    def call(self, inputs, training=True):
        x, a = inputs
        hx = self.dense_x(x)
        ha = self.dense_a(a)
        h = hx + ha
        h = self.net(h, training=training)
        return h
#%%
class zEncoder(K.models.Model):
    def __init__(self, latent_dim, **kwargs):
        super(zEncoder, self).__init__(**kwargs)
        self.dense_x = layers.Dense(256, activation='linear')
        self.dense_y = layers.Dense(256, activation='linear')
        self.dense_a = layers.Dense(256, activation='linear')
        self.net = K.Sequential(
            [
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dense(128, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.mean_layer = layers.Dense(latent_dim, activation='linear')
        self.logvar_layer = layers.Dense(latent_dim, activation='softplus')
    
    # @tf.function
    def call(self, inputs, training=True):
        x, y, a = inputs
        hx = self.dense_x(x)
        hy = self.dense_y(y)
        ha = self.dense_a(a)
        h = hx + hy + ha
        h = self.net(h, training=training)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar
#%%
class auxDecoder(K.models.Model):
    def __init__(self, a_dim, **kwargs):
        super(auxDecoder, self).__init__(**kwargs)
        self.dense_x = layers.Dense(256, activation='linear')
        self.dense_y = layers.Dense(256, activation='linear')
        self.dense_z = layers.Dense(256, activation='linear')
        self.net = K.Sequential(
            [
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dense(128, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.mean_layer = layers.Dense(a_dim, activation='linear')
        self.logvar_layer = layers.Dense(a_dim, activation='softplus')
    
    # @tf.function
    def call(self, inputs, training=True):
        z, y, x = inputs
        hx = self.dense_x(x)
        hy = self.dense_y(y)
        hz = self.dense_z(z)
        h = hx + hy + hz
        h = self.net(h, training=training)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar
#%%
class Decoder(K.models.Model):
    def __init__(self, activation='sigmoid', **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_z = layers.Dense(128, activation='linear')
        self.dense_y = layers.Dense(128, activation='linear')
        self.net = K.Sequential(
            [
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dense(256, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dense(784, activation=activation),
                layers.Reshape((28, 28, 1)),
            ]
        )
    
    # @tf.function
    def call(self, inputs, training=True):
        z, y = inputs
        hz = self.dense_z(z)
        hy = self.dense_y(y)
        h = hz + hy
        h = self.net(h, training=training)
        return h
#%%
class ADGM(K.models.Model):
    def __init__(self, 
                 args,
                 num_classes=10,
                 latent_dim=2, 
                 a_dim=2,
                 activation='sigmoid',
                 input_dim=(None, 28, 28, 1), 
                 **kwargs):
        super(ADGM, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.a_dim = a_dim
        self.input_dim = input_dim
        
        self.aux_encoder = auxEncoder(a_dim)
        self.aux_decoder = auxDecoder(a_dim)
        
        self.encoder = zEncoder(latent_dim)
        self.classifier = Classifier(num_classes)
        self.decoder = Decoder(activation)
    
    def encode(self, inputs, training=True):
        x, y = inputs
        x = layers.Flatten()(x)
        
        qa_mean, qa_logvar = self.aux_encoder(x, training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.a_dim))
        a = qa_mean + tf.math.exp(qa_logvar / 2.) * epsilon 
        
        mean, logvar = self.encoder([x, y, a], training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        return qa_mean, qa_logvar, a, mean, logvar, z
    
    def classify(self, x, training=True):
        x = layers.Flatten()(x)
        qa_mean, qa_logvar = self.aux_encoder(x, training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.a_dim))
        a = qa_mean + tf.math.exp(qa_logvar / 2.) * epsilon 
        prob = self.classifier([x, a], training=training)
        return prob
    
    def decode(self, z, y, training=True):
        xhat = self.decoder([z, y], training=training) 
        return xhat
        
    @tf.function
    def call(self, inputs, training=True):
        x, y = inputs
        x = layers.Flatten()(x)
        
        qa_mean, qa_logvar = self.aux_encoder(x, training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.a_dim))
        a = qa_mean + tf.math.exp(qa_logvar / 2.) * epsilon 
        
        mean, logvar = self.encoder([x, y, a], training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        
        xhat = self.decoder([z, y], training=training) 
        
        pa_mean, pa_logvar = self.aux_decoder([x, y, z], training=training)
        
        return mean, logvar, z, xhat, a, qa_mean, qa_logvar, pa_mean, pa_logvar
#%%
# class Classifier(K.models.Model):
#     def __init__(self, num_classes, a_dim, **kwargs):
#         super(Classifier, self).__init__(**kwargs)
#         self.a_dim = a_dim
#         self.nets = K.Sequential(
#             [
#                 layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same'), 
#                 layers.BatchNormalization(),
#                 layers.LeakyReLU(0.1),
                
#                 layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
#                 layers.SpatialDropout2D(rate=0.5),
                
#                 layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'), 
#                 layers.BatchNormalization(),
#                 layers.LeakyReLU(0.1),
                
#                 layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
#                 layers.SpatialDropout2D(rate=0.1),
                
#                 layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'), 
#                 layers.BatchNormalization(),
#                 layers.LeakyReLU(0.1),
                
#                 layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
#                 layers.SpatialDropout2D(rate=0.1),
                
#                 layers.GlobalAveragePooling2D(),
                
#                 layers.Dense(64, activation='linear'),
#                 layers.BatchNormalization(),
#                 layers.ReLU(),
#                 layers.Dense(num_classes, activation='softmax'),
#             ]
#         )
    
#     # @tf.function
#     def call(self, inputs, training=True):
#         x, a = inputs
#         a = tf.reshape(a, [-1, 1, 1, self.a_dim])
#         a = tf.tile(a, (1, tf.shape(x)[1], tf.shape(x)[2], 1))
#         h = tf.concat([x, a], axis=-1)
#         h = self.nets(h, training=training)
#         return h
#%%