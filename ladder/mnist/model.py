#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class Encoder(K.models.Model):
    def __init__(self, hidden_dim, latent_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.net = K.Sequential(
            [
                layers.Dense(hidden_dim, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        
        self.mean_layer = layers.Dense(latent_dim, activation='linear')
        self.logvar_layer = layers.Dense(latent_dim, activation='softplus')
    
    # @tf.function
    def call(self, x, training=True):
        d = self.net(x, training=training) # deterministic
        mean = self.mean_layer(d)
        logvar = self.logvar_layer(d)
        return mean, logvar
#%%
class Decoder(K.models.Model):
    def __init__(self, activation='sigmoid', name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.net = K.Sequential(
            [
                layers.Dense(128, activation='linear'),
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
    def call(self, x, training=True):
        h = self.net(x, training=training)
        return h
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
class LadderVAE(K.models.Model):
    def __init__(self, 
                 args,
                 num_classes=10,
                 z_dims = [32, 8, 2], 
                 activation='sigmoid',
                 input_dim=(None, 28, 28, 1), 
                 **kwargs):
        super(LadderVAE, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.h_dims = [4 * x for x in self.z_dims]
        self.input_dim = input_dim
        
        self.ladder_encoders = [Encoder(h, z) for h, z in zip(self.h_dims, self.z_dims)]
        self.ladder_decoders = [Encoder(h, z) for h, z in zip(self.h_dims[:-1][::-1], self.z_dims[:-1][::-1])]
        self.ladder_priors = [Encoder(h, z) for h, z in zip(self.h_dims[:-1][::-1], self.z_dims[:-1][::-1])]
        self.classifier = Classifier(num_classes)
        self.decoder = Decoder(activation)
    
    def encode(self, inputs, training=True):
        x, y = inputs
        
        # bottom-up of inference
        d = layers.Flatten()(x)
        bottomup = []
        for e in self.ladder_encoders:
            mean, logvar = e(tf.concat([d, y], axis=-1), training=training)
            epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mean)[-1]))
            z = mean + tf.math.exp(logvar / 2.) * epsilon 
            d = mean
            bottomup.append((z, mean, logvar))
        
        # top-down of inference
        topdown = []
        for e in self.ladder_decoders:
            mean, logvar = e(tf.concat([z, y], axis=-1), training=training)
            epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mean)[-1]))
            z = mean + tf.math.exp(logvar / 2.) * epsilon 
            topdown.append((z, mean, logvar))
        
        # posterior distribution
        posterior = [bottomup[-1]]
        for (_, mean1, logvar1), (_, mean2, logvar2) in zip(bottomup[::-1][1:], topdown):
            precision1 = 1. / tf.math.exp(logvar1)
            precision2 = 1. / tf.math.exp(logvar2)
            mu = (mean1 * precision1 + mean2 * precision2) / (precision1 + precision2)
            logsigma = tf.math.log(1. / (precision1 + precision2))
            epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mu)[-1]))
            z = mu + tf.math.exp(logsigma / 2.) * epsilon 
            posterior.append((z, mu, logsigma))
        
        return posterior
    
    def classify(self, x, training=True):
        prob = self.classifier(x, training=training)
        return prob
    
    @tf.function
    def call(self, inputs, training=True):
        x, y = inputs
        
        # bottom-up of inference
        d = layers.Flatten()(x)
        bottomup = []
        for e in self.ladder_encoders:
            mean, logvar = e(tf.concat([d, y], axis=-1), training=training)
            epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mean)[-1]))
            z = mean + tf.math.exp(logvar / 2.) * epsilon 
            d = mean
            bottomup.append((z, mean, logvar))
        
        # top-down of inference
        topdown = []
        for e in self.ladder_decoders:
            mean, logvar = e(z, training=training)
            epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mean)[-1]))
            z = mean + tf.math.exp(logvar / 2.) * epsilon 
            topdown.append((z, mean, logvar))
        
        # posterior distribution
        posterior = [bottomup[-1]]
        for (_, mean1, logvar1), (_, mean2, logvar2) in zip(bottomup[::-1][1:], topdown):
            precision1 = 1. / tf.math.exp(logvar1)
            precision2 = 1. / tf.math.exp(logvar2)
            mu = (mean1 * precision1 + mean2 * precision2) / (precision1 + precision2)
            logsigma = tf.math.log(1. / (precision1 + precision2))
            epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mu)[-1]))
            z = mu + tf.math.exp(logsigma / 2.) * epsilon 
            posterior.append((z, mu, logsigma))
        
        # prior distribution
        prior = [(tf.zeros(tf.shape(posterior[0][0])), tf.zeros(tf.shape(posterior[0][0])))]
        for (z, _, _), e in zip(posterior[:-1], self.ladder_priors):
            mean, logvar = e(z, training=training)
            prior.append((mean, logvar))
        
        z = posterior[-1][0]
        xhat = self.decoder(tf.concat([z, y], axis=-1), training=training) 
        
        return posterior, prior, xhat
#%%
# z_dims = [32, 8, 2]
# h_dims = [4 * x for x in z_dims]
# ladder_encoders = [Encoder(h, z) for h, z in zip(h_dims, z_dims)]
# ladder_decoders = [Encoder(h, z) for h, z in zip(h_dims[:-1][::-1], z_dims[:-1][::-1])]
# ladder_priors = [Encoder(h, z) for h, z in zip(h_dims[:-1][::-1], z_dims[:-1][::-1])]

# x = tf.random.normal((4, 784))
# y = tf.random.normal((4, 10))

# d = x
# bottomup = []
# for e in ladder_encoders:
#     mean, logvar = e(tf.concat([d, y], axis=-1))
#     epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mean)[-1]))
#     z = mean + tf.math.exp(logvar / 2.) * epsilon 
#     d = mean
#     bottomup.append((z, mean, logvar))
# #%%
# topdown = []
# for e in ladder_decoders:
#     mean, logvar = e(tf.concat([z, y], axis=-1))
#     epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mean)[-1]))
#     z = mean + tf.math.exp(logvar / 2.) * epsilon 
#     topdown.append((z, mean, logvar))
# #%%
# posterior = [bottomup[-1]]
# for (_, mean1, logvar1), (_, mean2, logvar2) in zip(bottomup[::-1][1:], topdown):
#     precision1 = 1. / tf.math.exp(logvar1)
#     precision2 = 1. / tf.math.exp(logvar2)
#     mu = (mean1 * precision1 + mean2 * precision2) / (precision1 + precision2)
#     logsigma = tf.math.log(1. / (precision1 + precision2))
#     epsilon = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(mu)[-1]))
#     z = mu + tf.math.exp(logsigma / 2.) * epsilon 
#     posterior.append((z, mu, logsigma))
# #%%
# prior = [(tf.zeros(tf.shape(posterior[0][0])), tf.zeros(tf.shape(posterior[0][0])))]
# for (z, _, _), e in zip(posterior[:-1], ladder_priors):
#     mean, logvar = e(z)
#     prior.append((mean, logvar))
#%%