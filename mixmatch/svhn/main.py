#%%
import argparse
import os
# os.chdir(r'D:\semi\mixmatch') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/mixmatch') # main directory (repository)
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
from model import WideResNet
from mixmatch import mixmatch, semi_loss, linear_rampup, interleave, weight_decay, ema
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
#     tags=["svhn", "MixMatch"],
# )
#%%
def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=None, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='svhn',
                        help='dataset used for training')

    parser.add_argument('--epochs', type=int, default=1024, 
                        help='number of epochs, (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='examples per batch (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.002, 
                        help='learning_rate, (default: 0.01)')

    parser.add_argument('--labeled_examples', type=int, default=1000, 
                        help='number labeled examples (default: 1000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    parser.add_argument('--val_iteration', type=int, default=1024, 
                        help='number of iterations before validation (default: 1024)')
    parser.add_argument('--T', type=float, default=0.5, 
                        help='temperature sharpening ratio (default: 0.5)')
    parser.add_argument('--K', type=int, default=2, 
                        help='number of rounds of augmentation (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='param for sampling from Beta distribution (default: 0.75)')
    parser.add_argument('--lambda_u', type=int, default=75, 
                        help='multiplier for unlabeled loss (default: 100)')
    parser.add_argument('--rampup_length', type=int, default=1027,
                        help='rampup length for unlabelled loss multiplier (default: 16)')
    parser.add_argument('--weight_decay', type=float, default=0.02, 
                        help='decay rate for model vars (default: 0.02)')
    parser.add_argument('--ema_decay', type=float, default=0.999, 
                        help='ema decay for ema model vars (default: 0.999)')

    parser.add_argument('--depth', type=int, default=28, 
                        help='depth for WideResnet (default: 28)')
    parser.add_argument('--width', type=int, default=4, 
                        help='widen factor for WideResnet (default: 2)')
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='slope parameter for LeakyReLU (default: 0.1)')

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

    start_epoch = 0
    model_path = f'./assets/{current_time}'
    if not os.path.exists(f'{model_path}'):
        os.makedirs(f'{model_path}')
    
    save_path = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args['dataset'], save_path, args)
    #%%
    model = WideResNet(num_classes, depth=args['depth'], width=args['width'], slope=args['slope'])
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = K.optimizers.Adam(lr=args['learning_rate'])

    ema_model = WideResNet(num_classes, depth=args['depth'], width=args['width'], slope=args['slope'])
    ema_model.build(input_shape=(None, 32, 32, 3))
    ema_model.set_weights(model.get_weights())
    #%%
    test_accuracy_print = 0.
    args['T'] = tf.constant(args['T'])
    args['beta'] = tf.Variable(0., shape=())
    for epoch in range(start_epoch, args['epochs']):
        xe_loss, l2u_loss, total_loss, accuracy = train(datasetL, datasetU, model, ema_model, optimizer, epoch, args, test_accuracy_print)
        val_xe_loss, val_accuracy = validate(val_dataset, ema_model, epoch, args, split='Validation')
        test_xe_loss, test_accuracy = validate(test_dataset, ema_model, epoch, args, split='Test')
        
        step = args['val_iteration'] * (epoch + 1)
        # wandb.log({'(train) xe_loss': xe_loss.result().numpy()})
        # wandb.log({'(train) l2u_loss': l2u_loss.result().numpy()})
        # wandb.log({'(train) total_loss': total_loss.result().numpy()})
        # wandb.log({'(train) accuracy': accuracy.result().numpy()})
        
        # wandb.log({'(val) xe_loss': val_xe_loss.result().numpy()})
        # wandb.log({'(val) accuracy': val_accuracy.result().numpy()})
        
        # wandb.log({'(test) xe_loss': test_xe_loss.result().numpy()})
        # wandb.log({'(test) accuracy': test_accuracy.result().numpy()})
        
        test_accuracy_print = test_accuracy.result()

        # Reset metrics every epoch
        xe_loss.reset_states()
        l2u_loss.reset_states()
        total_loss.reset_states()
        accuracy.reset_states()
        val_xe_loss.reset_states()
        val_accuracy.reset_states()
        test_xe_loss.reset_states()
        test_accuracy.reset_states()
    #%%
    '''model & configurations save'''        
    ema_model.save_weights(model_path + '/model.h5', save_format="h5")

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
def train(datasetL, datasetU, model, ema_model, optimizer, epoch, args, test_accuracy_print):
    xe_loss_avg = tf.keras.metrics.Mean()
    l2u_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)

    iteratorL = iter(shuffle_and_batch(datasetL))
    iteratorU = iter(shuffle_and_batch(datasetU))

    progress_bar = tqdm.tqdm(range(args['val_iteration']), unit='batch')
    for batch_num in progress_bar:
        lambda_u = args['lambda_u'] * linear_rampup(epoch + batch_num/args['val_iteration'], args['rampup_length'])
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(shuffle_and_batch(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, _ = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(datasetU))
            imageU, _ = next(iteratorU)

        args['beta'].assign(np.random.beta(args['alpha'], args['alpha']))
        with tf.GradientTape() as tape:
            # run mixmatch
            XU, XUy = mixmatch(model, imageL, labelL, imageU, args['T'], args['K'], args['beta'])
            logits = [model(XU[0])]
            for batch in XU[1:]:
                logits.append(model(batch))
            logits = interleave(logits, args['batch_size'])
            logits_x = logits[0]
            logits_u = tf.concat(logits[1:], axis=0)

            # compute loss
            xe_loss, l2u_loss = semi_loss(XUy[:args['batch_size']], logits_x, XUy[args['batch_size']:], logits_u)
            total_loss = xe_loss + lambda_u * l2u_loss

        # compute gradients and run optimizer step
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema(model, ema_model, args['ema_decay'])
        weight_decay(model=model, decay_rate=args['weight_decay'] * args['learning_rate'])

        xe_loss_avg(xe_loss)
        l2u_loss_avg(l2u_loss)
        total_loss_avg(total_loss)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), model(tf.cast(imageL, dtype=tf.float32), training=False))

        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'XE Loss': f'{xe_loss_avg.result():.4f}',
            'L2U Loss': f'{l2u_loss_avg.result():.4f}',
            'WeightU': f'{lambda_u:.3f}',
            'Total Loss': f'{total_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}',
        })
    return xe_loss_avg, l2u_loss_avg, total_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, split):
    accuracy = tf.keras.metrics.Accuracy()
    xe_avg = tf.keras.metrics.Mean()

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        logits = model(image, training=False)
        xe_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
        xe_avg(xe_loss)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy(prediction, tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} XE Loss: {xe_avg.result():.4f}, {split} Accuracy: {accuracy.result():.3%}')

    return xe_avg, accuracy
#%%
if __name__ == '__main__':
    main()
#%%