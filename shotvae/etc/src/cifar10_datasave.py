#%%
'''
EXoN: EXplainable encoder Network
with CIFAR-10 dataset
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r'D:\EXoN')
#%%
PARAMS = {
    "data": 'cifar10',
    "class_num": 10,
    "channel": 3, 
    "activation": 'tanh',
}
#%%
'''data'''
(x_train, y_train), (_, _) = K.datasets.cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
#%%
labeled = 10000
np.random.seed(1)
# ensure that all classes are balanced 
lidx = np.concatenate([np.random.choice(np.where(y_train == i)[0], int(labeled / PARAMS['class_num']), replace=False) 
                        for i in range(PARAMS['class_num'])])
# np.random.shuffle(lidx)
uidx = np.array([x for x in np.arange(len(x_train)) if x not in lidx])
# np.random.shuffle(uidx)

x_train_U = x_train[uidx]
y_train_U = y_train_onehot[uidx]
x_train_L = x_train[lidx]
y_train_L = y_train_onehot[lidx]
#%%
data_dir = r'D:\cifar10_{}'.format(labeled)
#%%
'''train - total'''
for i in tqdm(range(len(x_train)), desc='train x generating: total'):
    np.save(data_dir + '/x_{}'.format(i), x_train[i, ...])
    
for i in tqdm(range(len(y_train_onehot)), desc='train y generating: total'):
    np.save(data_dir + '/y_{}'.format(i), y_train_onehot[i, ...])
#%%
'''train - labeled'''
for i in tqdm(range(len(x_train_L)), desc='train x generating: labeled'):
    np.save(data_dir + '/x_labeled_{}'.format(i), x_train_L[i, ...])

for i in tqdm(range(len(x_train_L)), desc='train y generating: labeled'):
    np.save(data_dir + '/y_labeled_{}'.format(i), y_train_L[i, ...])
#%%
'''train - unlabeled'''
for i in tqdm(range(len(x_train_U)), desc='train x generating: unlabeled'):
    np.save(data_dir + '/x_unlabeled_{}'.format(i), x_train_U[i, ...])

for i in tqdm(range(len(x_train_U)), desc='train y generating: unlabeled'):
    np.save(data_dir + '/y_unlabeled_{}'.format(i), y_train_U[i, ...])
#%%
# '''train - labeled'''
# batch_num = len(x_train_L) // PARAMS['batch_size'] + 1
# for i in tqdm(range(batch_num), desc='train x batch generating: labeled'):
#     np.save(data_dir + '/x_labeled_batch{}'.format(i), x_train_L[i*PARAMS['batch_size'] : (i+1)*PARAMS['batch_size'], ...])

# batch_num = len(x_train_L) // PARAMS['batch_size'] + 1
# for i in tqdm(range(batch_num), desc='train y batch generating: labeled'):
#     np.save(data_dir + '/y_labeled_batch{}'.format(i), y_train_L[i*PARAMS['batch_size'] : (i+1)*PARAMS['batch_size'], ...])
# #%%
# '''train - unlabeled'''
# batch_num = len(x_train_U) // PARAMS['batch_size'] + 1
# for i in tqdm(range(batch_num), desc='train x batch generating: unlabeled'):
#     np.save(data_dir + '/x_unlabeled_batch{}'.format(i), x_train_U[i*PARAMS['batch_size'] : (i+1)*PARAMS['batch_size'], ...])

# batch_num = len(x_train_U) // PARAMS['batch_size'] + 1
# for i in tqdm(range(batch_num), desc='train y batch generating: unlabeled'):
#     np.save(data_dir + '/y_unlabeled_batch{}'.format(i), y_train_U[i*PARAMS['batch_size'] : (i+1)*PARAMS['batch_size'], ...])
#%%
# np.load(data_dir + '/x_labeled_batch{}.npy'.format(i))
#%%