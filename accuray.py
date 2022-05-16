#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorboard as tb
import numpy as np
#%%
"""classification error"""
with open("accuracy.txt", "w") as file:
    
    for model in ['pi', 'mixmatch', 'plcb', 
                  'dgm', 'partedvae', 'shotvae']: # vat
        # dir = 'D:/semi/{}/logs/cifar10_4000'.format(model)
        dir = '/Users/anseunghwan/Documents/GitHub/semi/{}/logs/cifar10_4000'.format(model)
        file_list = [x for x in os.listdir(dir) if x not in ['.DS_Store', 'datasets', 'etc', 'warmup']]
        
        error = []
        for i in range(len(file_list)):
            path = dir + '/{}/test'.format(file_list[i])
            event_acc = EventAccumulator(path)
            event_acc.Reload()
            tag = 'accuracy'
            event_list = event_acc.Tensors(tag)
            value = tf.io.decode_raw(event_list[-1].tensor_proto.tensor_content, 
                                    event_list[-1].tensor_proto.dtype)
            error.append(100. * (1. - value.numpy()[0]))

        file.write("{} | mean: {:.3f}, std: {:.3f}\n\n".format(model, np.mean(error), np.std(error)))
#%%
"""inception score"""
#%%