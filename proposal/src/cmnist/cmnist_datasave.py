#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pprint import pprint
import os
import cv2
os.chdir(r'D:\cmnist')
# os.chdir('/Users/anseunghwan/Documents/GitHub/normalizing_flow')
#%%
PARAMS = {
    "batch_size": 64,
    "data": "cmnist",
    "class_num": 10,
}
#%%
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

'''colored mnist'''
PARAMS['data_dim'] = 32
PARAMS["channel"] = 3

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])

# image = x_train[0]
# plt.imshow(image)
# plt.show()
#%%
np.random.seed(1)
# def colored_mnist(image):
#     image = tf.image.resize(image, [PARAMS['data_dim'], PARAMS['data_dim']], method='nearest')
    
#     if tf.random.uniform((1, 1)) > 0.5:
#         # color
#         image = tf.cast(image, tf.float32) / 255.
#         color = np.random.uniform(0., 1., 3)
#         color = color / np.linalg.norm(color)
#         image = image * color[tf.newaxis, tf.newaxis, :]
#         # np.unique(image.numpy())
        
#         assert image.shape == (PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel'])
#         return image.numpy() * 2. - 1.
#     else:
#         # edge detection
#         image = cv2.Canny(image.numpy(), 10., 255.)
#         image[np.where(image > 0)] = 1.
#         image[np.where(image <= 0)] = 0.

#         # color
#         color = np.random.uniform(0., 1., 3)
#         color = color / np.linalg.norm(color)
#         image = image[..., tf.newaxis] * color[tf.newaxis, tf.newaxis, :]
        
#         # width
#         kernel = np.ones((1, 1))
#         image = cv2.dilate(image, kernel)

#         assert image.shape == (PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel'])
#         return image * 2. - 1.

color_list = [
    (255, 0, 0), # red 
    (255, 0, 128), # rose 
    (255, 0, 255), # magenta 
    (128, 0, 255), # violet
    (0, 0, 255), # blue 
    (0, 128, 255), # azure 
    (0, 255, 255), # cyan 
    (0, 255, 128), # spring green 
    (0, 255, 0), # green
    (128, 255, 0), # chartreuse 
    (255, 255, 0), # yellow 
    (255, 128, 0), # orange
]

def colored_mnist(image):
    
    image = tf.image.resize(image, [32, 32], method='nearest')
    
    if tf.random.uniform((1, 1)) > 0.5:
        # color
        image = tf.cast(image, tf.float32) / 255.
        # color = np.random.uniform(0., 1., 3)
        # color = color / np.linalg.norm(color)
        color = np.array(color_list[np.random.choice(range(len(color_list)), 1)[0]]) / 255.
        image = image * color[tf.newaxis, tf.newaxis, :]
        return image
    else:
        # edge detection
        image = cv2.Canny(image.numpy(), 10., 255.)
        image[np.where(image > 0)] = 1.
        image[np.where(image <= 0)] = 0.
        # color
        # color = np.random.uniform(0., 1., 3)
        # color = color / np.linalg.norm(color)
        color = np.array(color_list[np.random.choice(range(len(color_list)), 1)[0]]) / 255.
        image = image[..., tf.newaxis] * color[tf.newaxis, tf.newaxis, :]
        # width
        kernel = np.ones((1, 1))
        image = cv2.dilate(image, kernel)
        return image
#%%
cx_train = []
for i in tqdm(range(len(x_train)), desc='generating train colored mnist'):
    cx_train.append(colored_mnist(x_train[i]))
cx_train = np.array(cx_train)

plt.imshow(cx_train[3])
plt.show()

cx_test = []
for i in tqdm(range(len(x_test)), desc='generating test colored mnist'):
    cx_test.append(colored_mnist(x_test[i]))
cx_test = np.array(cx_test)
#%%
'''train'''
for i in tqdm(range(len(x_train)), desc='train x generating'):
    np.save('./train/x_{}'.format(i), cx_train[i, ...])
    
for i in tqdm(range(len(y_train_onehot)), desc='train y generating'):
    np.save('./train/y_{}'.format(i), y_train_onehot[i, ...])
#%%
'''test'''
for i in tqdm(range(len(x_test)), desc='test x generating'):
    np.save('./test/x_{}'.format(i), cx_test[i, ...])
    
for i in tqdm(range(len(y_test_onehot)), desc='test y generating'):
    np.save('./test/y_{}'.format(i), y_test_onehot[i, ...])
#%%