#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class ConvLayer(K.layers.Layer):
    def __init__(self, filter_size, kernel_size, strides, name="ConvLayer", **kwargs):
        super(ConvLayer, self).__init__(name=name, **kwargs)
        self.conv2d = layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding='same')
        self.norm = layers.BatchNormalization()

    @tf.function
    def call(self, x, training=True):
        h = self.conv2d(x)
        h = self.norm(h, training=training)
        h = tf.nn.relu(h)
        return h
#%%
class zEncoder(K.models.Model):
    def __init__(self, latent_dim, name="Encoder", **kwargs):
        super(zEncoder, self).__init__(name=name, **kwargs)
        self.conv = [
            ConvLayer(16, 5, 2), # 16x16
            ConvLayer(32, 5, 2), # 8x8
            ConvLayer(64, 3, 2), # 4x4
            ConvLayer(128, 3, 2) # 2x2
            ]
        self.dense = layers.Dense(256)
        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.last = layers.Dense(latent_dim)
    
    @tf.function
    def call(self, x, training=True):
        h = x
        for i in range(len(self.conv)):
            h = self.conv[i](h, training=training)
        h = layers.Flatten()(h)
        h = self.relu(self.norm(self.dense(h), training=training))
        h = self.last(h)
        return h
#%%
class cEncoder(K.models.Model):
    def __init__(self, num_classes, name="Encoder", **kwargs):
        super(cEncoder, self).__init__(name=name, **kwargs)
        self.conv = [
            ConvLayer(16, 5, 2), # 16x16
            ConvLayer(32, 5, 2), # 8x8
            ConvLayer(64, 3, 2), # 4x4
            ConvLayer(128, 3, 2) # 2x2
            ] 
        self.dense = layers.Dense(256)
        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.last = layers.Dense(num_classes)
        
    def call(self, x, training=True):
        h = x
        for i in range(len(self.conv)):
            h = self.conv[i](h, training=training)
        h = layers.Flatten()(h)
        h = self.relu(self.norm(self.dense(h), training=training))
        h = self.last(h)
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
        
        self.reshape1 = layers.Reshape((4, 4, latent_dim // 16))
        self.dense = layers.Dense(16, use_bias=False)
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
        h1 = self.reshape1(z)
        h2 = self.reshape2(self.dense(prob))
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
                 activation='sigmoid',
                 input_shape=(None, 32, 32, 3), 
                 name='AutoEncoder', **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        
        self.z_encoder = zEncoder(latent_dim)
        self.c_encoder = cEncoder(num_classes) 
        self.Decoder = Decoder(latent_dim, output_channel, activation)
        
    def z_encode(self, x, training=False):
        z = self.z_encoder(x, training=training)
        return z
    
    def c_encode(self, x, training=False):
        c = self.c_encoder(x, training=training)
        return c
    
    def decode(self, z, y, training=False):
        return self.Decoder(z, y, training=training) 
        
    @tf.function
    def call(self, x, training=True):
        z = self.z_encoder(x, training=training)
        c = self.c_encoder(x, training=training)
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
        self.ae = AutoEncoder(num_classes, args['latent_dim'])
        self.prior = Prior(args, num_classes)
        self.prior.build_graph()
        
    def call(self, x, training=True):
        z, c, prob, xhat = self.ae(x, training=training)
        z_ = tf.stop_gradient(z)
        c_ = tf.stop_gradient(c)
        nf_args = self.prior(z_, c_)
        return [[z, c, prob, xhat], nf_args]
#%%