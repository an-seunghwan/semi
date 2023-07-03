#%%
import argparse
import os

# os.chdir(r'D:\semi\pi') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/pi') # main directory (repository)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

import tqdm
import yaml
import io
import matplotlib.pyplot as plt
import random as python_random

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import CNN
from utils import augment
#%%
# import sys
# import subprocess
# try:
#     import wandb
# except:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
#     with open("./wandb_api.txt", "r") as f:
#         key = f.readlines()
#     subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
#     import wandb

# run = wandb.init(
#     project="EXoN", 
#     entity="anseunghwan",
#     tags=["svhn", "Pi-model"],
# )
#%%
import ast
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
#%%
def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='svhn',
                        help='dataset used for training')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results (ex. generating color MNIST)')
    parser.add_argument('--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--labeled-batch-size', default=50, type=int,
                        metavar='N', help='mini-batch size of labeled dataset')

    '''SSL Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=300, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--labeled_examples', type=int, default=1000, 
                        help='number labeled examples (default: 1000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--initial_beta1', default=0.9, type=float, 
                        help='initial beta_1 value of optimizer')
    parser.add_argument('--final_beta1', default=0.5, type=float, 
                        help='final beta_1 value of optimizer')
    parser.add_argument('--ramp_up_period', default=80, type=int, 
                        help='ramp-up period of loss function')
    parser.add_argument('--ramp_down_period', default=50, type=int, 
                        help='ramp-down period')
    parser.add_argument('--weight_max', default=100, type=float, 
                        help='related to unsupervised loss component')
    
    parser.add_argument('--augmentation_flag', default=True, type=bool, 
                        help='Data augmentation')
    parser.add_argument('--trans_range', default=2, type=int, 
                        help='random_translation_range')

    '''Configuration'''
    parser.add_argument('--config_path', type=str, default=None, 
                        help='path to yaml config file, overwrites args')
    
    return parser.parse_args()
#%%
def load_config(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, args['config_path'])    
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args
#%%
def main():
    #%%
    '''argparse to dictionary'''
    args = vars(get_args())
    # '''argparse debugging'''
    # args = vars(parser.parse_args(args=[]))
    # wandb.config.update(args)
    #%%
    np.random.seed(args["seed"])
    python_random.seed(args["seed"])
    tf.random.set_seed(args["seed"])
    #%%
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    model_path = f'./assets/{current_time}'
    if not os.path.exists(f'{model_path}'):
        os.makedirs(f'{model_path}')
    
    save_path = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args['dataset'], save_path, args)
    total_length = sum(1 for _ in datasetU)
    #%%
    model = CNN(num_classes)
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    
    test_accuracy_print = 0.
    
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['learning_rate'])
    #%%
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        T = epoch / args['ramp_up_period']
        ramp_up = tf.math.exp(-5. * (tf.math.maximum(0., 1. - T)) ** 2)
        T = (args['epochs'] - epoch) / args['ramp_down_period']
        ramp_down = tf.math.exp(-12.5 * (tf.math.maximum(0., 1. - T)) ** 2)
        
        '''unsupervised loss weight'''
        if epoch == 0:
            loss_weight = 0
        else:
            loss_weight = ramp_up * ((args['weight_max'] * args['labeled_examples']) / (50000. * num_classes))
        
        '''learning rate schedule'''
        optimizer.lr = ramp_up * ramp_down * args['learning_rate']
        optimizer.beta_1 = ramp_down * args['initial_beta1'] + (1. - ramp_down) * args['final_beta1']
        
        loss, ce_loss, u_loss, accuracy = train(datasetL, datasetU, model, optimizer, epoch, args, loss_weight, num_classes, total_length, test_accuracy_print)
        val_ce_loss, val_accuracy = validate(val_dataset, model, epoch, args, split='Validation')
        test_ce_loss, test_accuracy = validate(test_dataset, model, epoch, args, split='Test')
        
        # wandb.log({'(train) loss': loss.result().numpy()})
        # wandb.log({'(train) ce_loss': ce_loss.result().numpy()})
        # wandb.log({'(train) u_loss': u_loss.result().numpy()})
        # wandb.log({'(train) accuracy': accuracy.result().numpy()})
        
        # wandb.log({'(val) ce_loss': val_ce_loss.result().numpy()})
        # wandb.log({'(val) accuracy': val_accuracy.result().numpy()})
        
        # wandb.log({'(test) ce_loss': test_ce_loss.result().numpy()})
        # wandb.log({'(test) accuracy': test_accuracy.result().numpy()})
        
        test_accuracy_print = test_accuracy.result()

        # Reset metrics every epoch
        loss.reset_states()
        ce_loss.reset_states()
        u_loss.reset_states()
        accuracy.reset_states()
        val_ce_loss.reset_states()
        val_accuracy.reset_states()
        test_ce_loss.reset_states()
        test_accuracy.reset_states()
    #%%
    '''model & configurations save'''        
    model.save_weights(model_path + '/model.h5', save_format="h5")

    with open(model_path + '/args.txt', "w") as f:
        for key, value, in args.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    
    # artifact = wandb.Artifact(
    #     f'{args["dataset"]}_Pi_model', 
    #     type='model',
    #     metadata=args) # description=""
    # artifact.add_file(model_path + '/args.txt')
    # artifact.add_file(model_path + '/model.h5')
    # artifact.add_file('./main.py')
    # wandb.log_artifact(artifact)
    # #%%
    # wandb.config.update(args, allow_val_change=True)
    # wandb.run.finish()
#%%
def train(datasetL, datasetU, model, optimizer, epoch, args, loss_weight, num_classes, total_length, test_accuracy_print):
    loss_avg = tf.keras.metrics.Mean()
    ce_loss_avg = tf.keras.metrics.Mean()
    u_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    shuffle_and_batch2 = lambda dataset: dataset.shuffle(buffer_size=int(1e4)).batch(batch_size=args['labeled_batch_size'], drop_remainder=True)
    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'] - args['labeled_batch_size'], drop_remainder=True)

    iteratorL = iter(shuffle_and_batch2(datasetL))
    iteratorU = iter(shuffle_and_batch(datasetU))
        
    iteration = total_length // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(shuffle_and_batch2(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, _ = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(datasetU))
            imageU, _ = next(iteratorU)
        
        if args['augmentation_flag']:
            imageL = augment(imageL, args['trans_range'])
            imageU = augment(imageU, args['trans_range'])
            
        image = tf.concat([imageL, imageU], axis=0)
        
        '''normalization'''
        channel_stats = dict(mean=tf.reshape(tf.cast(np.array([0.4376821, 0.4437697, 0.47280442]), tf.float32), (1, 1, 1, 3)),
                             std=tf.reshape(tf.cast(np.array([0.19803012, 0.20101562, 0.19703614]), tf.float32), (1, 1, 1, 3)))
        image -= channel_stats['mean']
        image /= channel_stats['std']
        
        with tf.GradientTape(persistent=True) as tape:
            pred1 = model(image)
            pred2 = model(image)
            
            # supervised
            predL = tf.gather(pred1, tf.range(args['labeled_batch_size']))
            ce_loss = - tf.reduce_mean(tf.reduce_sum(labelL * tf.math.log(tf.clip_by_value(predL, 1e-10, 1.0)), axis=-1))
            
            # unsupervised
            u_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(pred1 - pred2), axis=-1))
            
            loss = ce_loss + loss_weight * u_loss
            
        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        
        loss_avg(loss)
        ce_loss_avg(ce_loss)
        u_loss_avg(u_loss)
        probL = model(image, noise=False, training=False)
        probL = tf.gather(probL, tf.range(args['labeled_batch_size']))
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)

        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'CE_Loss': f'{ce_loss_avg.result():.4f}',
            'U_Loss': f'{u_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}',
        })
        
    return loss_avg, ce_loss_avg, u_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, split):
    ce_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.Accuracy()

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        channel_stats = dict(mean=tf.reshape(tf.cast(np.array([0.4376821, 0.4437697, 0.47280442]), tf.float32), (1, 1, 1, 3)),
                             std=tf.reshape(tf.cast(np.array([0.19803012, 0.20101562, 0.19703614]), tf.float32), (1, 1, 1, 3)))
        image -= channel_stats['mean']
        image /= channel_stats['std']
        pred = model(image, noise=False, training=False)
        ce_loss = - tf.reduce_mean(tf.reduce_sum(label * tf.math.log(tf.clip_by_value(pred, 1e-10, 1.0)), axis=-1))
        ce_loss_avg(ce_loss)
        accuracy(tf.argmax(pred, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split}, CE: {ce_loss_avg.result():.4f},  Accuracy: {accuracy.result():.3%}')
    
    return ce_loss_avg, accuracy
#%%
if __name__ == '__main__':
    main()
#%%