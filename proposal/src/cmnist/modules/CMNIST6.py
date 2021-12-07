#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class ConvLayer(K.layers.Layer):
    def __init__(self, params, filter_size, kernel_size, strides, name="ConvLayer", **kwargs):
        super(ConvLayer, self).__init__(name=name, **kwargs)
        self.params = params
        self.conv2d = layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding='same')
        self.norm = layers.BatchNormalization()

    def call(self, x, training=True):
        h = self.conv2d(x)
        h = self.norm(h, training=training)
        h = tf.nn.relu(h)
        return h
#%%
class zEncoder(K.models.Model):
    def __init__(self, params, name="Encoder", **kwargs):
        super(zEncoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.conv = [
            ConvLayer(self.params, 16, 5, 2), # 16x16
            ConvLayer(self.params, 32, 5, 2), # 8x8
            ConvLayer(self.params, 64, 3, 2), # 4x4
            ConvLayer(self.params, 128, 3, 2) # 2x2
            ]
        self.dense = layers.Dense(256, activation='linear')
        self.norm = layers.BatchNormalization()
        self.last = layers.Dense(self.params['z_dim'], activation='linear')
        
    def call(self, x, training=True):
        h = x
        for i in range(len(self.conv)):
            h = self.conv[i](h, training=training)
        h = layers.Flatten()(h)
        h = tf.nn.relu(self.norm(self.dense(h), training=training))
        h = self.last(h)
        return h
#%%
class cEncoder(K.models.Model):
    def __init__(self, params, name="Encoder", **kwargs):
        super(cEncoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.conv = [
            ConvLayer(self.params, 16, 5, 2), # 16x16
            ConvLayer(self.params, 32, 5, 2), # 8x8
            ConvLayer(self.params, 64, 3, 2), # 4x4
            ConvLayer(self.params, 128, 3, 2) # 2x2
            ] 
        self.dense = layers.Dense(256, activation='linear')
        self.norm = layers.BatchNormalization()
        self.last = layers.Dense(self.params['c_dim'], activation='linear')
        
    def call(self, x, training=True):
        h = x
        for i in range(len(self.conv)):
            h = self.conv[i](h, training=training)
        h = layers.Flatten()(h)
        h = tf.nn.relu(self.norm(self.dense(h), training=training))
        h = self.last(h)
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
    def __init__(self, params, name="AutoEncoder", **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.zEncoder = zEncoder(self.params)
        self.cEncoder = cEncoder(self.params)
        self.Decoder = build_generator(self.params)
        
    def call(self, x, training=True):
        z = self.zEncoder(x, training=training)
        c = self.cEncoder(x, training=training)
        prob = tf.nn.softmax(c)
        xhat = self.Decoder([z, prob], training=training)
        
        return [z, c, prob, xhat]
#%%
# class CouplingLayerBN(K.layers.Layer):
#     def __init__(self, input_dim, epsilon=1e-8, axis=-1, **kwargs):
#         super(CouplingLayerBN, self).__init__(**kwargs)
        
#         self.input_dim = input_dim
#         self.epsilon = epsilon
#         self.axis = axis

#         self.gamma = self.add_weight(shape=(self.input_dim,),
#                                     initializer=K.initializers.Zeros(),
#                                     name='{}_gamma'.format(self.name),
#                                     trainable=True)
#         self.beta = self.add_weight(shape=(self.input_dim,),
#                                     initializer=K.initializers.Ones(),
#                                     name='{}_beta'.format(self.name),
#                                     trainable=True)

#     def call(self, x):
#         mean, std = tf.nn.moments(x, axes=0, keepdims=True)
#         x = (x - mean) / (std + self.epsilon)
#         x = x * tf.math.exp(self.gamma) + self.beta
#         return x, std

#     def inverse(self, x):
#         mean, std = tf.nn.moments(x, axes=0, keepdims=True)
#         x = (x - self.beta) * tf.math.exp(- self.gamma) * (std + self.epsilon) + mean
#         return x
#%%
class zCouplingLayer(K.models.Model):
    def __init__(self, params, embedding_dim, output_dim, activation, name='zCouplingLayer', **kwargs):
        super(zCouplingLayer, self).__init__(name=name, **kwargs)
        
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
class cCouplingLayer(K.models.Model):
    def __init__(self, params, embedding_dim, output_dim, activation, name='cCouplingLayer', **kwargs):
        super(cCouplingLayer, self).__init__(name=name, **kwargs)
        
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
        
    def call(self, x, y):
        x = tf.concat([x, y], axis=-1)
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
        
        self.s = [zCouplingLayer(self.params, self.params['z_embedding_dim'], self.params['z_nf_dim'], activation='tanh')
                    for _ in range(self.params['K1'])]
        self.t = [zCouplingLayer(self.params, self.params['z_embedding_dim'], self.params['z_nf_dim'], activation='linear')
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
        
        self.s = [cCouplingLayer(self.params, self.params['c_embedding_dim'], self.params['c_nf_dim'], activation='tanh')
                   for _ in range(self.params['K2'])]
        self.t = [cCouplingLayer(self.params, self.params['c_embedding_dim'], self.params['c_nf_dim'], activation='linear')
                   for _ in range(self.params['K2'])]
        
    def inverse(self, x, condition):
        for i in reversed(range(self.params['K2'])):
            x_masked = tf.boolean_mask(x, self.mask[i], axis=1)
            x = (
                x * self.mask[i]
                + (1 - self.mask[i])
                * ((x - tf.repeat(self.t[i](x_masked, condition), 2, axis=1))
                   *
                   tf.repeat(tf.math.exp(- self.s[i](x_masked, condition)), 2, axis=1))
            )
        return x
        
    def call(self, x, condition, sum_log_abs_det_jacobians=None):
        if sum_log_abs_det_jacobians is None:
            sum_log_abs_det_jacobians = 0
        log_abs_det_jacobian = 0
        
        for i in range(self.params['K2']):
            x_masked = tf.boolean_mask(x, self.mask[i], axis=1)
            x = (
                x * self.mask[i]
                + (1 - self.mask[i])
                * (x * tf.repeat(tf.math.exp(self.s[i](x_masked, condition)), 2, axis=1) 
                   + 
                   tf.repeat(self.t[i](x_masked, condition), 2, axis=1))
            )
            
            log_abs_det_jacobian += tf.reduce_sum(self.s[i](x_masked, condition), axis=-1)
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
        dummy_y = tf.random.normal((1, self.params['class_num']))
        _ = self(dummy_z, dummy_c, dummy_y)
        return print('Graph is built!')
    
    def zflow(self, x):
        return self.zNF.inverse(x)
    
    def cflow(self, x, y):
        return self.cNF.inverse(x, y)
    
    def call(self, z, c, y):
        z_ = tf.stop_gradient(z)
        z_sg, sum_log_abs_det_jacobians1 = self.zNF(z_)
        
        c_ = tf.stop_gradient(c)
        c_sg, sum_log_abs_det_jacobians2 = self.cNF(c_, y)
    
        return [z_sg, sum_log_abs_det_jacobians1, c_sg, sum_log_abs_det_jacobians2]
#%%
class DeterministicVAE(K.models.Model):
    def __init__(self, params, name="DeterministicVAE", **kwargs):
        super(DeterministicVAE, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.AE = AutoEncoder(self.params)
        self.Prior = Prior(self.params)
        
    def call(self, x, y, training=True):
        z, c, prob, xhat = self.AE(x, training=training)
        
        prior_args = self.Prior(z, c, y)
        
        return [[z, c, prob, xhat], prior_args]
#%%