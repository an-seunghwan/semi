#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
#%%
class FeatureExtractor(K.models.Model):
    def __init__(self, hidden_dim, name="FeatureExtractor", **kwargs):
        super(FeatureExtractor, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.nChannels = [32, 64, 64]
        
        self.net = K.Sequential(
            [
                layers.Conv2D(filters=self.nChannels[0], kernel_size=4, strides=2, padding='same'), # 14x14
                layers.ReLU(),
                
                layers.Conv2D(filters=self.nChannels[1], kernel_size=4, strides=2, padding='same'), # 7x7
                layers.ReLU(),
                
                layers.Conv2D(filters=self.nChannels[2], kernel_size=4, strides=2, padding='same'), # 4x4
                layers.LeakyReLU(alpha=0.1),
                
                layers.Flatten(),
                layers.Dense(self.hidden_dim),
                layers.LeakyReLU(alpha=0.1),
            ]
        )
        
    @tf.function
    def call(self, x, training=True):
        hidden = self.net(x, training=training)
        return hidden
#%%
# e = FeatureExtractor(256)
# e.build((10, 28, 28, 1))
# e.net.summary()
#%%
class Decoder(K.models.Model):
    def __init__(self, hidden_dim, activation, channel, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.nChannels = [32, 32]
        
        self.net = K.Sequential(
            [
                layers.Dense(hidden_dim),
                layers.ReLU(),
                
                layers.Dense(7 * 7 * 64),
                layers.ReLU(),
                layers.Reshape((7, 7, 64)),
                
                layers.Conv2DTranspose(filters=self.nChannels[0], kernel_size=4, strides=2, padding='same'),
                layers.ReLU(),
                
                layers.Conv2DTranspose(filters=self.nChannels[1], kernel_size=4, strides=2, padding='same'),
                layers.ReLU(),
                
                layers.Conv2DTranspose(filters=channel, kernel_size=4, strides=1, padding='same', activation=activation)
            ]
        )
    
    @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        return h
#%%
# e = Decoder(256, 'sigmoid', 1)
# e.build((10, 4))
# e.net.summary()
#%%
class VAE(K.models.Model):
    def __init__(self, 
                 num_classes=10,
                 latent_dim=6, 
                 hidden_dim=256,
                 u_dim=10,
                 output_channel=1, 
                 temperature=0.67,
                 sigmoid_coef=1,
                 activation='sigmoid',
                 input_dim=(None, 28, 28, 1), 
                 name='VAE', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.u_dim = u_dim
        self.output_channel = output_channel
        self.input_dim = input_dim
        self.temperature = temperature
        self.sigmoid_coef = sigmoid_coef
        self.activation = activation
        
        self.feature_extractor = FeatureExtractor(self.hidden_dim)
        
        self.z_mean_layer = layers.Dense(latent_dim, activation='linear')
        self.z_logvar_layer = layers.Dense(latent_dim, activation='linear')
        
        self.h_to_c_logit = layers.Dense(num_classes, activation='linear') # h to c logit
        self.c_to_a_logit = layers.Dense(self.hidden_dim, activation='linear') # c to a logit
        
        self.u_mean_layer = layers.Dense(u_dim, activation='linear')
        self.u_logvar_layer = layers.Dense(u_dim, activation='linear')
        
        self.decoder = Decoder(self.hidden_dim, self.activation, self.output_channel)
        
        self.u_prior_means = self.add_weight(shape=(num_classes, u_dim),
                                            initializer='random_normal',
                                            trainable=True,
                                            name='u_prior_means')
        self.u_prior_logvars_before_tanh = self.add_weight(shape=(num_classes, u_dim),
                                                            initializer='random_normal',
                                                            trainable=True,
                                                            name='u_prior_logvars_before_tanh')
    
    @property
    def u_prior_logvars(self):
        return 2. * tf.nn.tanh(self.u_prior_logvars_before_tanh) - 1.
    
    def sample_gumbel(self, shape): 
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + 1e-8) + 1e-8)

    def gumbel_softmax_sample(self, log_prob, training=True): 
        y = log_prob + self.sample_gumbel(tf.shape(log_prob))
        if training:
            y = tf.nn.softmax(y / self.temperature)
        else:
            y = tf.cast(tf.equal(y, tf.math.reduce_max(y, axis=1, keepdims=True)), y.dtype)
        return y
    
    def _sigmoid(self, x, training=True):
        if not training or self.sigmoid_coef > 8.:
            return tf.nn.sigmoid(8. * x)
        if self.sigmoid_coef < 8.:
            self.sigmoid_coef += 2e-4
        return tf.nn.sigmoid(self.sigmoid_coef * x)
    
    def classify(self, x, training=True):
        hidden = self.feature_extractor(x, training=training)
        c_logit = self.h_to_c_logit(hidden)
        return tf.nn.softmax(c_logit, axis=-1)
    
    def encode(self, x, training=True):
        hidden = self.feature_extractor(x, training=training)
        
        # class independent
        z_mean = self.z_mean_layer(hidden)
        z_logvar = self.z_logvar_layer(hidden)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = z_mean + tf.math.exp(z_logvar / 2.) * epsilon 
        
        # class related
        c_logit = self.h_to_c_logit(hidden)
        onehot_c = self.gumbel_softmax_sample(c_logit, training=training)
        a_logit = self.c_to_a_logit(onehot_c)
        a = self._sigmoid(a_logit)
        hidden_a = hidden * a
        u_mean = self.u_mean_layer(hidden_a)
        u_logvar = self.u_logvar_layer(hidden_a)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.u_dim))
        u = u_mean + tf.math.exp(u_logvar / 2.) * epsilon 
        return z_mean, z_logvar, z, u_mean, u_logvar, u
    
    @tf.function
    def call(self, x, training=True):
        hidden = self.feature_extractor(x, training=training)
        
        # class independent
        z_mean = self.z_mean_layer(hidden)
        z_logvar = self.z_logvar_layer(hidden)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = z_mean + tf.math.exp(z_logvar / 2.) * epsilon 
        
        # class related
        c_logit = self.h_to_c_logit(hidden)
        onehot_c = self.gumbel_softmax_sample(c_logit, training=training)
        a_logit = self.c_to_a_logit(onehot_c)
        a = self._sigmoid(a_logit)
        hidden_a = hidden * a
        u_mean = self.u_mean_layer(hidden_a)
        u_logvar = self.u_logvar_layer(hidden_a)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.u_dim))
        u = u_mean + tf.math.exp(u_logvar / 2.) * epsilon 
        
        xhat = self.decoder(tf.concat([z, u], axis=-1), training=training) 
        return z_mean, z_logvar, z, c_logit, u_mean, u_logvar, u, xhat
#%%
# model = VAE()
# model.build((10, 28, 28, 1))
# %%
# x = tf.random.normal((16, 28, 28, 1))
# outputs = model(x)
# [t.shape for t in outputs]
# model.u_prior_means.shape
# model.u_prior_logvars.shape
# model.feature_extractor.summary()
# model.feature_extractor.trainable_variables + model.h_to_c_logit.trainable_variables
#%%