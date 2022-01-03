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
        self.Decoder = Decoder(latent_dim, output_channel, activation)
        
    def z_encode(self, x, training=False):
        h = self.FeatureExtractor(x, training=training)
        z = self.z_layer(h)
        return z
    
    def c_encode(self, x, training=False):
        h = self.FeatureExtractor(x, training=training)
        c = self.c_layer(h)
        return c
    
    def decode(self, z, y, training=False):
        return self.Decoder(z, y, training=training) 
        
    @tf.function
    def call(self, x, training=True):
        h = self.FeatureExtractor(x, training=training)
        z = self.z_layer(h)
        c = self.c_layer(h)
        prob = tf.nn.softmax(c, axis=-1)
        xhat = self.Decoder(z, prob, training=training) 
        return z, c, prob, xhat
#%%
class CouplingLayer(K.models.Model):
    def __init__(self, 
                 activation, 
                 embedding_dim, 
                 output_dim, 
                 coupling_MLP_num,
                 reg=0.01,
                 name='CouplingLayer', **kwargs):
        super(CouplingLayer, self).__init__(name=name, **kwargs)
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dense = [
            layers.Dense(self.embedding_dim, activation='relu', kernel_regularizer=K.regularizers.l2(reg)) 
            for _ in range(coupling_MLP_num)
            ] + [
            layers.Dense(self.output_dim, activation=self.activation, kernel_regularizer=K.regularizers.l2(reg))
            ]
    
    def call(self, x):
        for d in self.dense:
            x = d(x)
        return x
#%%
def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2
#%%
class NormalizingFlow(K.models.Model):
    def __init__(self, 
                 latent_dim,
                 embedding_dim,
                 mask,
                 coupling_MLP_num,
                 K,
                 name='NormalizingFlow', **kwargs):
        super(NormalizingFlow, self).__init__(name=name, **kwargs)
        
        self.K = K
        if mask == 'checkerboard':
            self.mask = [checkerboard((latent_dim, )), 
                        1 - checkerboard((latent_dim, ))] * (K // 2)
        elif mask == 'half':
            self.mask = [np.array([1] * (latent_dim // 2) + [0] * (latent_dim // 2)),
                        np.array([0] * (latent_dim // 2) + [1] * (latent_dim // 2))] * (K // 2)
        
        output_dim = latent_dim // 2
        self.s = [CouplingLayer('tanh', embedding_dim, output_dim, coupling_MLP_num) for _ in range(K)]
        self.t = [CouplingLayer('linear', embedding_dim, output_dim, coupling_MLP_num) for _ in range(K)]
        
    def inverse(self, x):
        for i in reversed(range(self.K)):
            x_masked = tf.boolean_mask(x, self.mask[i], axis=1)
            x = (
                x * self.mask[i]
                + (1 - self.mask[i])
                * ((x - tf.repeat(self.t[i](x_masked), 2, axis=1))
                   *
                   tf.repeat(tf.math.exp(- self.s[i](x_masked)), 2, axis=1))
            )
        return x
    
    def call(self, x, sum_log_abs_det_jacobians=None):
        if sum_log_abs_det_jacobians is None:
            sum_log_abs_det_jacobians = 0
        log_abs_det_jacobian = 0
        
        for i in range(self.K):
            x_masked = tf.boolean_mask(x, self.mask[i], axis=1)
            x = (
                x * self.mask[i]
                + (1 - self.mask[i])
                * (x * tf.repeat(tf.math.exp(self.s[i](x_masked)), 2, axis=1) 
                   + 
                   tf.repeat(self.t[i](x_masked), 2, axis=1))
            )
            
            log_abs_det_jacobian += tf.reduce_sum(self.s[i](x_masked), axis=-1)
        sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians
#%%
class Prior(K.models.Model):
    def __init__(self, 
                 args,
                 num_classes,
                 name="Prior", **kwargs):
        super(Prior, self).__init__(name=name, **kwargs)
        self.args = args
        self.num_classes = num_classes
        
        self.zNF = NormalizingFlow(args['latent_dim'], args['z_emb'], args['z_mask'], args['coupling_MLP_num'], args['K1'])
        self.cNF = NormalizingFlow(num_classes, args['c_emb'], args['c_mask'], args['coupling_MLP_num'], args['K2'])
    
    def build_graph(self):
        '''
        build model manually due to masked coupling layer
        '''
        dummy_z = tf.random.normal((1, self.args['latent_dim']))
        dummy_c = tf.random.normal((1, self.num_classes))
        _ = self(dummy_z, dummy_c)
        return print('Graph is built!')
    
    def zflow(self, x):
        return self.zNF.inverse(x)
    
    def cflow(self, x):
        return self.cNF.inverse(x)
    
    def call(self, z, c):
        z_sg, sum_log_abs_det_jacobians1 = self.zNF(z)
        c_sg, sum_log_abs_det_jacobians2 = self.cNF(c)
        return [z_sg, sum_log_abs_det_jacobians1, c_sg, sum_log_abs_det_jacobians2]
#%%
class VAE(K.models.Model):
    def __init__(self, args, num_classes, name="VAE", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.args = args
        self.ae = AutoEncoder(num_classes, args['depth'], args['width'], args['slope'], args['latent_dim'])
        self.prior = Prior(args, num_classes)
        self.prior.build_graph()
    
    @tf.function
    def call(self, x, training=True):
        z, c, prob, xhat = self.ae(x, training=training)
        z_ = tf.stop_gradient(z)
        c_ = tf.stop_gradient(c)
        nf_args = self.prior(z_, c_)
        return [[z, c, prob, xhat], nf_args]
#%%
class AuxiliaryClassifier(K.models.Model):
    def __init__(self, 
                 num_classes,
                 name='AuxiliaryClassifier', **kwargs):
        super(AuxiliaryClassifier, self).__init__(name=name, **kwargs)
        
        self.dense = [layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(num_classes, activation='softmax')]
        
    def call(self, x):
        for d in self.dense:
            x = d(x)
        return x
#%%