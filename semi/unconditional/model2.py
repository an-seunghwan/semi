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
        self.conv2 = layers.Conv2D(filters=filter_out, kernel_size=3, strides=1, 
                                    padding='same', use_bias=False)
        
        self.downsample = (filter_in != filter_out)
        if self.downsample:
            self.shortcut = layers.Conv2D(filters=filter_out, kernel_size=1, strides=strides, 
                                        padding='same', use_bias=False)

    @tf.function
    def call(self, x, training=True):
        if self.downsample:
            x = self.relu1(self.norm1(x, training=training))
            h = self.relu2(self.norm2(self.conv1(x), training=training))
        else:
            h = self.relu1(self.norm1(x, training=training))
            h = self.relu2(self.norm2(self.conv1(h), training=training))
        h = self.conv2(h)
        if self.downsample:
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
        return K.models.Sequential(units)
    
    @tf.function
    def call(self, x, training=True):
        x = self.units(x, training=training)
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
        
        self.conv = layers.Conv2D(filters=self.nChannels[0], kernel_size=3, strides=1, 
                                    padding='same', use_bias=False)
        
        self.block1 = ResidualBlock(self.n_units, self.nChannels[0], self.nChannels[1], ResidualUnit, 1)
        self.block2 = ResidualBlock(self.n_units, self.nChannels[1], self.nChannels[2], ResidualUnit, 2)
        self.block3 = ResidualBlock(self.n_units, self.nChannels[2], self.nChannels[3], ResidualUnit, 2)
        
        self.norm = layers.BatchNormalization()
        self.relu = layers.LeakyReLU(alpha=slope)
        self.pooling = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(num_classes)
    
    @tf.function
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
                 depth=28,
                 width=2,
                 slope=0.1,
                 latent_dim=128, 
                 output_channel=3, 
                 activation='sigmoid',
                 input_shape=(None, 32, 32, 3), 
                 name='AutoEncoder', **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        
        self.FeatureExtractor = WideResNet(num_classes, depth, width, slope, input_shape)
        self.z_layer = layers.Dense(latent_dim) 
        self.c_layer = layers.Dense(num_classes) 
        self.decoder = Decoder(latent_dim, output_channel, activation)
        
    def z_encode(self, x, training=False):
        h = self.FeatureExtractor(x, training=training)
        z = self.z_layer(h)
        return z
    
    def c_encode(self, x, training=False):
        h = self.FeatureExtractor(x, training=training)
        c = self.c_layer(h)
        return c
    
    def decode(self, z, y, training=False):
        return self.decoder(z, y, training=training) 
        
    @tf.function
    def call(self, x, training=True):
        h = self.FeatureExtractor(x, training=training)
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
        self.scale = self.add_weight(shape=(n_units, ),
                                    initializer='zeros',
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
        self.perm = tf.Variable(tf.random.shuffle(tf.range(start=0, limit=n_units, dtype=tf.int32)),
                                trainable=False)
        
    @tf.function
    def call(self, x):
        return tf.gather(x, self.perm, axis=-1)
    
    @tf.function
    def reverse(self, x):
        return tf.gather(x, tf.argsort(self.perm), axis=-1)
#%%
class NormalizingFlow(K.models.Model):
    def __init__(self, 
                 latent_dim,
                 hidden_dim,
                 n_blocks,
                 name='NormalizingFlow', **kwargs):
        super(NormalizingFlow, self).__init__(name=name, **kwargs)
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
        out = self.affine_layers[-1](x)
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