#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class BasicBlock(K.layers.Layer):
    def __init__(self, 
                 params,
                 in_filter_size, 
                 out_filter_size,
                 strides, 
                 slope=0.0,
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.params = params
        
        self.slope = slope
        self.norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filters=out_filter_size, kernel_size=3, strides=strides, 
                                    padding='same', use_bias=False, kernel_regularizer=K.regularizers.l2(self.params['weight_decay']))
        self.norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=out_filter_size, kernel_size=3, strides=1, 
                                    padding='same', use_bias=False, kernel_regularizer=K.regularizers.l2(self.params['weight_decay']))
        
        self.equalInOut = (in_filter_size == out_filter_size)
        if not self.equalInOut:
            self.conv3 = layers.Conv2D(filters=out_filter_size, kernel_size=1, strides=strides, 
                                        padding='same', use_bias=False, kernel_regularizer=K.regularizers.l2(self.params['weight_decay']))

    def call(self, x, training=True):
        if not self.equalInOut:
            x = tf.nn.leaky_relu(self.norm1(x, training=training), alpha=self.slope)
            h = tf.nn.leaky_relu(self.norm2(self.conv1(x), training=training), alpha=self.slope)
        else:
            h = tf.nn.leaky_relu(self.norm1(x, training=training), alpha=self.slope)
            h = tf.nn.leaky_relu(self.norm2(self.conv1(h), training=training), alpha=self.slope)
        h = self.conv2(h)
        if not self.equalInOut:
            h = h + self.conv3(x)
        else:
            h = h + x
        return h
#%%
class WideResNet(K.models.Model):
    def __init__(self, params, name="WideResNet", **kwargs):
        super(WideResNet, self).__init__(name=name, **kwargs)
        self.params = params
        
        assert (self.params['depth'] - 4) % 6 == 0
        self.block_depth = (self.params['depth'] - 4) // 6
        self.nChannels = [self.params['decoder_feature'], 
                          self.params['decoder_feature'] * self.params['widen_factor'], 
                          2 * self.params['decoder_feature'] * self.params['widen_factor'], 
                          4 * self.params['decoder_feature'] * self.params['widen_factor']]
        
        # preprocess (small_input = True)
        self.conv = layers.Conv2D(filters=self.nChannels[0], kernel_size=3, strides=1, 
                                    padding='same', use_bias=False, kernel_regularizer=K.regularizers.l2(self.params['weight_decay']))
        
        # output: 28x28x16
        self.block1 = K.models.Sequential([BasicBlock(self.params, self.nChannels[0], self.nChannels[1], strides=1, slope=self.params['slope'])] + \
                                            [BasicBlock(self.params, self.nChannels[1], self.nChannels[1], strides=1, slope=self.params['slope'])])
        
        # output: 14x14x32
        self.block2 = K.models.Sequential([BasicBlock(self.params, self.nChannels[1], self.nChannels[2], strides=2, slope=self.params['slope'])] + \
                                            [BasicBlock(self.params, self.nChannels[2], self.nChannels[2], strides=1, slope=self.params['slope'])])
        
        # output: 7x7x64
        self.block3 = K.models.Sequential([BasicBlock(self.params, self.nChannels[2], self.nChannels[3], strides=2, slope=self.params['slope'])] + \
                                            [BasicBlock(self.params, self.nChannels[3], self.nChannels[3], strides=1, slope=self.params['slope'])])
        
        self.norm = layers.BatchNormalization()
        self.pooling = layers.GlobalAveragePooling2D()
        
    def call(self, x, training=True):
        h = self.conv(x)
        h = self.block1(h, training=training)
        h = self.block2(h, training=training)
        h = self.block3(h, training=training)
        h = tf.nn.leaky_relu(self.norm(h, training=training), alpha=self.params['slope'])
        h = self.pooling(h)
        return h
#%%
def build_generator(PARAMS):
    z = layers.Input(PARAMS['z_dim'])
    c = layers.Input(PARAMS['c_dim'])
    
    hc = layers.Dense(16, use_bias=False)(c)
    hc = layers.Reshape((4, 4, 1))(hc)
    
    h = layers.Reshape((4, 4, PARAMS['z_dim'] // 16))(z)
    h = layers.Concatenate()([h, hc])
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    dims = [128, 64, 32]
    skip = h
    for i in range(3):
        skip = layers.Conv2DTranspose(filters = dims[i], kernel_size = 5, strides = 2, padding = 'same', use_bias=False)(skip)
        skip = layers.BatchNormalization()(skip)
        skip = layers.ReLU()(skip)
        
        h = layers.Conv2D(filters = dims[i], kernel_size = 3, strides = 1, padding = 'same', use_bias=False)(skip)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        
        h = layers.Conv2D(filters = dims[i], kernel_size = 3, strides = 1, padding = 'same', use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        
        skip = h + skip
    
    h = layers.Conv2D(filters = 3, kernel_size = 1, strides = 1, padding='same', activation='tanh')(skip)
    
    G = K.models.Model([z, c], h)
    # G.summary()
    
    return G
#%%
class AutoEncoder(K.models.Model):
    def __init__(self, params, name='AutoEncoder', **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.FeatureExtractor = WideResNet(self.params)
        self.z_layer = layers.Dense(self.params['z_dim'], 
                                    kernel_regularizer=K.regularizers.l2(self.params['weight_decay'])) 
        self.c_layer = layers.Dense(self.params['class_num'], 
                                    kernel_regularizer=K.regularizers.l2(self.params['weight_decay'])) 
        self.Decoder = build_generator(self.params)
    
    def get_latent(self, x, training=True):
        h = self.FeatureExtractor(x, training=training)
        z = self.z_layer(h)
        return z
    
    def get_prob(self, x, training=True):
        h = self.FeatureExtractor(x, training=training)
        c = self.c_layer(h)
        prob = tf.nn.softmax(c)
        return prob
    
    def call(self, x, training=True):
        h = self.FeatureExtractor(x, training=training)
        z = self.z_layer(h)
        c = self.c_layer(h)
        prob = tf.nn.softmax(c)
        xhat = self.Decoder([z, prob], training=training) 
        return z, c, prob, xhat
#%%
class CouplingLayer(K.models.Model):
    def __init__(self, params, embedding_dim, output_dim, activation, name='CouplingLayer', **kwargs):
        super(CouplingLayer, self).__init__(name=name, **kwargs)
        
        self.params = params
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dense = [
            layers.Dense(self.embedding_dim, activation='relu', kernel_regularizer=K.regularizers.l2(self.params['reg'])) 
            for _ in range(self.params['coupling_MLP_num'])
            ] + [
            layers.Dense(self.output_dim, activation=self.activation, kernel_regularizer=K.regularizers.l2(self.params['reg']))
            ]
        
    def call(self, x):
        for d in self.dense:
            x = d(x)
        return x
#%%
def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2
#%%
class zNormalizingFlow(K.models.Model):
    def __init__(self, params, name='zNormalizingFlow', **kwargs):
        super(zNormalizingFlow, self).__init__(name=name, **kwargs)
        
        self.params = params
        if self.params['z_mask'] == 'checkerboard':
            self.mask = [checkerboard((self.params['z_dim'], )), 
                        1 - checkerboard((self.params['z_dim'], ))] * (self.params['K1'] // 2)
        elif self.params['z_mask'] == 'half':
            self.mask = [np.array([1] * (self.params['z_dim'] // 2) + [0] * (self.params['z_dim'] // 2)),
                        np.array([0] * (self.params['z_dim'] // 2) + [1] * (self.params['z_dim'] // 2))] * (self.params['K1'] // 2)
        
        self.s = [CouplingLayer(self.params, self.params['z_embedding_dim'], self.params['z_nf_dim'], activation='tanh')
                    for _ in range(self.params['K1'])]
        self.t = [CouplingLayer(self.params, self.params['z_embedding_dim'], self.params['z_nf_dim'], activation='linear')
                    for _ in range(self.params['K1'])]
        
    def inverse(self, x):
        for i in reversed(range(self.params['K1'])):
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
        
        for i in range(self.params['K1']):
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
class cNormalizingFlow(K.models.Model):
    def __init__(self, params, name='cNormalizingFlow', **kwargs):
        super(cNormalizingFlow, self).__init__(name=name, **kwargs)
        
        self.params = params
        if self.params['c_mask'] == 'checkerboard':
            self.mask = [checkerboard((self.params['c_dim'], )), 
                        1 - checkerboard((self.params['c_dim'], ))] * (self.params['K2'] // 2)
        elif self.params['c_mask'] == 'half':
            self.mask = [np.array([1] * (self.params['c_dim'] // 2) + [0] * (self.params['c_dim'] // 2)),
                        np.array([0] * (self.params['c_dim'] // 2) + [1] * (self.params['c_dim'] // 2))] * (self.params['K2'] // 2)
        
        self.s = [CouplingLayer(self.params, self.params['c_embedding_dim'], self.params['c_nf_dim'], activation='tanh')
                   for _ in range(self.params['K2'])]
        self.t = [CouplingLayer(self.params, self.params['c_embedding_dim'], self.params['c_nf_dim'], activation='linear')
                   for _ in range(self.params['K2'])]
        
    def inverse(self, x):
        for i in reversed(range(self.params['K2'])):
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
        
        for i in range(self.params['K2']):
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
    def __init__(self, params, name="Prior", **kwargs):
        super(Prior, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.zNF = zNormalizingFlow(self.params)
        self.cNF = cNormalizingFlow(self.params)
    
    def build_graph(self):
        '''
        build model manually due to masked coupling layer
        '''
        dummy_z = tf.random.normal((1, self.params['z_dim']))
        dummy_c = tf.random.normal((1, self.params['c_dim']))
        _ = self(dummy_z, dummy_c)
        return print('Graph is built!')
    
    def zflow(self, x):
        return self.zNF.inverse(x)
    
    def cflow(self, x, y):
        return self.cNF.inverse(x)
    
    def call(self, z, c):
        z_sg, sum_log_abs_det_jacobians1 = self.zNF(z)
        c_sg, sum_log_abs_det_jacobians2 = self.cNF(c)
        return [z_sg, sum_log_abs_det_jacobians1, c_sg, sum_log_abs_det_jacobians2]
#%%
class DeterministicVAE(K.models.Model):
    def __init__(self, params, name="DeterministicVAE", **kwargs):
        super(DeterministicVAE, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.AE = AutoEncoder(self.params)
        self.Prior = Prior(self.params)
        
    def call(self, x, training=True):
        z, c, prob, xhat = self.AE(x, training=training)
        z_ = tf.stop_gradient(z)
        c_ = tf.stop_gradient(c)
        prior_args = self.Prior(z_, c_)
        return [[z, c, prob, xhat], prior_args]
#%%