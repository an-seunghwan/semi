#%%
'''
220121: permutation tf.gather -> tf.one_hot matmul
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class ConvLayer(K.layers.Layer):
    def __init__(self, filter_size, kernel_size, strides, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.conv2d = layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding='same')
        self.norm = layers.BatchNormalization()

    def call(self, x, training=True):
        h = self.conv2d(x)
        h = self.norm(h, training=training)
        h = tf.nn.relu(h)
        return h
#%%
class FeatureExtractor(K.models.Model):
    def __init__(self, 
                 input_shape=(None, 32, 32, 3),
                 name="FeatureExtractor", 
                 **kwargs):
        super(FeatureExtractor, self).__init__(input_shape, name=name, **kwargs)
        
        self.units = K.Sequential(
            [
                ConvLayer(16, 5, 2), # 16x16
                ConvLayer(32, 5, 2), # 8x8
                ConvLayer(64, 3, 2), # 4x4
                ConvLayer(128, 3, 2), # 2x2
                layers.Flatten(),
                layers.Dense(256, activation='linear'),
                layers.BatchNormalization(),
            ]    
        )
        
    @tf.function
    def call(self, x, training=True):
        h = self.units(x, training=training)
        return h
#%%
class Decoder(K.models.Model):
    def __init__(self, 
                 latent_dim,
                 output_channel, 
                 activation, 
                 name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.num_feature = 32
        
        self.latent_dim = latent_dim
        if self.latent_dim < 16:
            self.dense1 = layers.Dense(16, use_bias=False)
            self.reshape1 = layers.Reshape((4, 4, 1))
        else:
            self.reshape1 = layers.Reshape((4, 4, latent_dim // 16))
        self.dense2 = layers.Dense(16, use_bias=False)
        self.reshape2 = layers.Reshape((4, 4, 1))
        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.units_skip, self.units = self._build_unit()
        self.conv = layers.Conv2DTranspose(filters = output_channel, kernel_size = 1, strides = 1, 
                                            activation=activation,
                                            padding = 'same', use_bias=False)
    
    def _build_unit(self):
        dims = [self.num_feature * d for d in [4, 2, 1]]
        units_skip = []
        for i in range(len(dims)):
            units_skip.append(
                K.Sequential(
                    [
                        layers.Conv2DTranspose(filters = dims[i], kernel_size = 5, strides = 2, 
                                                padding = 'same', use_bias=False),
                        layers.BatchNormalization(),
                        layers.ReLU()
                    ]
                )
            )
        units = []
        for i in range(len(dims)):
            units.append(
                K.Sequential(
                    [
                        layers.Conv2DTranspose(filters = dims[i], kernel_size = 3, strides = 1, 
                                                padding = 'same', use_bias=False),
                        layers.BatchNormalization(),
                        layers.ReLU(),
                        layers.Conv2DTranspose(filters = dims[i], kernel_size = 3, strides = 1, 
                                                padding = 'same', use_bias=False),
                        layers.BatchNormalization(),
                        layers.ReLU(),
                    ]
                )
            )
        return units_skip, units
    
    @tf.function
    def call(self, z, prob, training=True):
        if self.latent_dim < 16:
            h1 = self.reshape1(self.dense1(z))
        else:
            h1 = self.reshape1(z)
        h2 = self.reshape2(self.dense2(prob))
        h = tf.concat([h1, h2], axis=-1)
        h = self.relu(self.norm(h))
        
        skip = h
        for i in range(len(self.units_skip)):
            skip = self.units_skip[i](skip, training=training)    
            h = self.units[i](skip, training=training)
            skip += h
        h = self.conv(skip)
        return h
#%%
class AutoEncoder(K.models.Model):
    def __init__(self, 
                 num_classes=10,
                 latent_dim=128, 
                 output_channel=3, 
                 activation='tanh',
                 input_shape=(None, 32, 32, 3), 
                 name='AutoEncoder', **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        
        self.feature_extractor = FeatureExtractor(input_shape)
        self.z_layer = layers.Dense(latent_dim) 
        self.c_layer = layers.Dense(num_classes) 
        self.decoder = Decoder(latent_dim, output_channel, activation)
        
    def z_encode(self, x, training=True):
        h = self.feature_extractor(x, training=training)
        z = self.z_layer(h)
        return z
    
    def c_encode(self, x, training=True):
        h = self.feature_extractor(x, training=training)
        c = self.c_layer(h)
        return c
    
    def decode(self, z, y, training=True):
        return self.decoder(z, y, training=training) 
        
    @tf.function
    def call(self, x, training=True):
        h = self.feature_extractor(x, training=training)
        z = self.z_layer(h)
        c = self.c_layer(h)
        prob = tf.nn.softmax(c, axis=-1)
        xhat = self.decoder(z, prob, training=training) 
        return z, c, prob, xhat
#%%
class Scale(K.layers.Layer):
    def __init__(self,
                 n_units,
                 **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.scale = tf.Variable(tf.zeros((n_units, )),
                                name='scale',
                                trainable=True)
    
    @tf.function
    def call(self, x):
        return x * tf.math.exp(self.scale * 3.)
#%%
class AffineCoupling(K.layers.Layer):
    def __init__(self,
                 n_units,
                 hidden_dim,
                 **kwargs):
        super(AffineCoupling, self).__init__(**kwargs)
        self.net = K.Sequential(
                        [
                            layers.Dense(n_units // 2, activation="relu"),
                            layers.Dense(hidden_dim, activation="relu"),
                            layers.Dense(n_units, activation="linear"),
                            Scale(n_units)
                        ]
                    )
    
    @tf.function
    def call(self, x):
        z1, z2 = tf.split(x, num_or_size_splits=2, axis=-1)
        
        log_s, t = tf.split(self.net(z1), num_or_size_splits=2, axis=-1)
        s = tf.nn.sigmoid(log_s + 2.)
        
        y1 = z1
        y2 = (z2 + t) * s
        y = tf.concat([y1, y2], axis=-1)
        logdet = tf.reduce_sum(tf.math.log(s), axis=-1)
        return y, logdet
    
    @tf.function
    def reverse(self, y):
        y1, y2 = tf.split(y, num_or_size_splits=2, axis=-1)
        
        log_s, t = tf.split(self.net(y1), num_or_size_splits=2, axis=-1)
        s = tf.nn.sigmoid(log_s + 2.)
        
        z1 = y1
        z2 = y2 / s - t
        z = tf.concat([z1, z2], axis=-1)
        return z
#%%
class Permuatation(K.layers.Layer):
    def __init__(self,
                 n_units,
                 **kwargs):
        super(Permuatation, self).__init__(**kwargs)
        self.n_units = n_units
        self.perm = tf.Variable(tf.random.shuffle(tf.range(start=0, limit=n_units, dtype=tf.int32)),
                                name='permutation',
                                trainable=False)
        
    @tf.function
    def call(self, x):
        # return tf.gather(x, self.perm, axis=-1)
        return tf.matmul(x, tf.one_hot(self.perm, depth=self.n_units))
    
    @tf.function
    def reverse(self, x):
        # return tf.gather(x, tf.argsort(self.perm), axis=-1)
        return tf.matmul(x, tf.one_hot(tf.argsort(self.perm), depth=self.n_units))
#%%
class NormalizingFlow(K.models.Model):
    def __init__(self, 
                 latent_dim,
                 hidden_dim,
                 n_blocks,
                 **kwargs):
        super(NormalizingFlow, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        
        self.affine_layers = [AffineCoupling(latent_dim, hidden_dim) for _ in range(n_blocks)]
        self.permutations = [Permuatation(latent_dim) for _ in range(n_blocks - 1)]
        
    @tf.function
    def call(self, x):
        out = x
        logdets = 0.
        for i in range(self.n_blocks - 1):
            out, logdet = self.affine_layers[i](out)
            out = self.permutations[i](out)
            logdets += logdet
        out, logdet = self.affine_layers[-1](out)
        logdets += logdet
        return out, logdets
    
    @tf.function
    def reverse(self, x):
        out = self.affine_layers[-1].reverse(x)
        for i in range(self.n_blocks - 1):
            out = self.permutations[-1-i].reverse(out)
            out = self.affine_layers[-2-i].reverse(out)
        return out
#%%
# net = NormalizingFlow(20, 64, 4)
# out, logdet = net(tf.random.normal((2, 20)))
#%%
class Prior(K.models.Model):
    def __init__(self, 
                 args,
                 num_classes,
                 name="Prior", **kwargs):
        super(Prior, self).__init__(name=name, **kwargs)
        self.args = args
        self.num_classes = num_classes
        
        self.zNF = NormalizingFlow(args['latent_dim'], args['z_hidden_dim'], args['z_n_blocks'])
        self.cNF = NormalizingFlow(num_classes, args['c_hidden_dim'], args['c_n_blocks'])
    
    # def build_graph(self):
    #     '''
    #     build model manually due to masked coupling layer
    #     '''
    #     dummy_z = tf.random.normal((1, self.args['latent_dim']))
    #     dummy_c = tf.random.normal((1, self.num_classes))
    #     _ = self(dummy_z, dummy_c)
    #     return print('Graph is built!')
    
    def zflow(self, x):
        return self.zNF.reverse(x)
    
    def cflow(self, x):
        return self.cNF.reverse(x)
    
    def call(self, z, c):
        z_, logdet1 = self.zNF(z)
        c_, logdet2 = self.cNF(c)
        return [z_, logdet1, c_, logdet2]
#%%
class VAE(K.models.Model):
    def __init__(self, args, num_classes, name="VAE", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.args = args
        self.ae = AutoEncoder(num_classes, args['depth'], args['width'], args['slope'], args['latent_dim'])
        self.prior = Prior(args, num_classes)
        # self.prior.build_graph()
    
    @tf.function
    def call(self, x, training=True):
        z, c, prob, xhat = self.ae(x, training=training)
        z_ = tf.stop_gradient(z)
        c_ = tf.stop_gradient(c)
        nf_args = self.prior(z_, c_)
        return [[z, c, prob, xhat], nf_args]
#%%