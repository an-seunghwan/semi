#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
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
        self.nChannels = [16, 16*self.params['widen_factor'], 32*self.params['widen_factor'], 64*self.params['widen_factor']]
        
        # preprocess (small_input = True)
        self.conv = layers.Conv2D(filters=self.nChannels[0], kernel_size=3, strides=1, 
                                    padding='same', use_bias=False, kernel_regularizer=K.regularizers.l2(self.params['weight_decay']))
        
        # output: 32x32x32
        self.block1 = K.models.Sequential([BasicBlock(self.params, self.nChannels[0], self.nChannels[1], strides=1, slope=self.params['slope'])] + \
                                            [BasicBlock(self.params, self.nChannels[1], self.nChannels[1], strides=1, slope=self.params['slope'])])
        
        # output: 16x16x64
        self.block2 = K.models.Sequential([BasicBlock(self.params, self.nChannels[1], self.nChannels[2], strides=2, slope=self.params['slope'])] + \
                                            [BasicBlock(self.params, self.nChannels[2], self.nChannels[2], strides=1, slope=self.params['slope'])])
        
        # output: 8x8x128
        self.block3 = K.models.Sequential([BasicBlock(self.params, self.nChannels[2], self.nChannels[3], strides=2, slope=self.params['slope'])] + \
                                            [BasicBlock(self.params, self.nChannels[3], self.nChannels[3], strides=1, slope=self.params['slope'])])
        
        self.norm = layers.BatchNormalization()
        
    def call(self, x, training=True):
        h = self.conv(x)
        h = self.block1(h, training=training)
        h = self.block2(h, training=training)
        h = self.block3(h, training=training)
        h = tf.nn.leaky_relu(self.norm(h, training=training), alpha=self.params['slope'])
        h = layers.Flatten()(h)
        return h
#%%
class Decoder(K.models.Model):
    def __init__(self, params, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.params = params
        self.num_feature = 64
        self.model = []
        dims = [self.num_feature * d for d in [16, 8, 4, 2]]
        for i in range(len(dims)):
            self.model.append(layers.Conv2DTranspose(filters = dims[i], kernel_size = 5, strides = 2, 
                                                     padding = 'same', use_bias=False, kernel_regularizer=K.regularizers.l2(self.params['weight_decay'])))
            self.model.append(layers.BatchNormalization())
            self.model.append(layers.ReLU())
        self.model.append(layers.Conv2DTranspose(filters = self.params['channel'], kernel_size = 4, strides = 2, 
                                                 activation=self.params['activation'],
                                                 padding = 'same', use_bias=False, kernel_regularizer=K.regularizers.l2(self.params['weight_decay'])))
        
    def call(self, x, training=True):
        h = x[:, tf.newaxis, tf.newaxis, :]
        for i in range(len(self.model)):
            h = self.model[i](h, training=training)    
        return h
#%%
class VAE(K.models.Model):
    def __init__(self, params, name='VAE', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.FeatureExtractor = WideResNet(self.params)
        self.mean_layer = layers.Dense(self.params['latent_dim'], 
                                       kernel_regularizer=K.regularizers.l2(self.params['weight_decay'])) 
        self.logsigma_layer = layers.Dense(self.params['latent_dim'], 
                                         kernel_regularizer=K.regularizers.l2(self.params['weight_decay'])) 
        self.prob_layer = layers.Dense(self.params['class_num'],
                                       kernel_regularizer=K.regularizers.l2(self.params['weight_decay'])) 
        self.Decoder = Decoder(self.params)
        
    def sample_gumbel(self, shape): 
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + 1e-8) + 1e-8)

    def gumbel_max_sample(self, log_prob): 
        y = log_prob + self.sample_gumbel(tf.shape(log_prob))
        y = tf.nn.softmax(y / self.params['temperature'])
        if self.params['hard']:
            y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y
    
    def call(self, x, label=None, training=True):
        h = self.FeatureExtractor(x, training=training)
        mean = self.mean_layer(h)
        log_sigma = self.logsigma_layer(h)
        if label is not None:
            log_prob = label
            '''NOT Gumbel-Softmax'''
            y = label
        else:
            log_prob = tf.nn.log_softmax(self.prob_layer(h))
            y = self.gumbel_max_sample(log_prob)
        
        noise = tf.random.normal((x.shape[0], self.params["latent_dim"]))
        z = mean + tf.math.exp(log_sigma) * noise 
        
        xhat = self.Decoder(tf.concat([z, y], axis=-1), training=training) 
        return mean, log_sigma, log_prob, z, y, xhat
#%%