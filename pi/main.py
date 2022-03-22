#%%
import argparse
import os

os.chdir(r'D:\semi\pi') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/pi') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm
import yaml
import io
import matplotlib.pyplot as plt

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import CNN
from utils import augment
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

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra, cmnist)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results (ex. generating color MNIST)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')

    '''SSL Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=300, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning_rate', default=0.003, type=float,
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
    
    # parser.add_argument('--weight_norm_flag', default=True, type=bool,
    #                     help='Weight normalization is applied. Otherwise Batch normalization is applied')
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
    '''argparse to dictionary'''
    args = vars(get_args())
    # '''argparse debugging'''
    # args = vars(parser.parse_args(args=['--config_path', 'configs/cmnist_100.yaml']))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    total_length = sum(1 for _ in datasetU)
    
    model = CNN(num_classes)
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['lr'])
    
    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    for epoch in range(args['start_epoch'], args['epochs']):
        
        T = epoch / args['ramp_up_period']
        ramp_up = tf.math.exp(-5. * (tf.math.maximum(0., 1. - T)) ** 2)
        T = (args['epochs'] - epoch) / args['ramp_down_period']
        ramp_down = tf.math.exp(-12.5 * (tf.math.maximum(0., 1. - T)) ** 2)
        
        '''unsupervised loss weight'''
        if epoch == 0:
            loss_weight = 0
        else:
            loss_weight = ramp_up * (args['weight_max'] * args['labeled_examples'] / 50000.)
        
        '''learning rate schedule'''
        optimizer.lr = ramp_up * ramp_down * args['lr']
        optimizer.beta_1 = ramp_down * args['initial_beta1'] + (1. - ramp_down) * args['final_beta1']
        
        loss, ce_loss, u_loss, accuracy = train(datasetL, datasetU, model, optimizer, epoch, args, loss_weight, num_classes, total_length)
        val_ce_loss, val_accuracy = validate(val_dataset, model, epoch, args, split='Validation')
        test_ce_loss, test_accuracy = validate(test_dataset, model, epoch, args, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('ce_loss', ce_loss.result(), step=epoch)
            tf.summary.scalar('u_loss', u_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('ce_loss', val_ce_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar('ce_loss', test_ce_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        # Reset metrics every epoch
        loss.reset_states()
        ce_loss.reset_states()
        u_loss.reset_states()
        accuracy.reset_states()
        val_ce_loss.reset_states()
        val_accuracy.reset_states()
        test_ce_loss.reset_states()
        test_accuracy.reset_states()

    '''model & configurations save'''        
    # weight name for saving
    for i, w in enumerate(model.variables):
        split_name = w.name.split('/')
        if len(split_name) == 1:
            new_name = split_name[0] + '_' + str(i)    
        else:
            new_name = split_name[0] + '_' + str(i) + '/' + split_name[1] + '_' + str(i)
        model.variables[i]._handle_name = new_name
    
    model_path = f'{log_path}/{current_time}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save_weights(model_path + '/model_{}.h5'.format(current_time), save_format="h5")

    with open(model_path + '/args_{}.txt'.format(current_time), "w") as f:
        for key, value, in args.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
#%%
def train(datasetL, datasetU, model, optimizer, epoch, args, loss_weight, num_classes, total_length):
    loss_avg = tf.keras.metrics.Mean()
    ce_loss_avg = tf.keras.metrics.Mean()
    u_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    shuffle_and_batch2 = lambda dataset: dataset.shuffle(buffer_size=int(1e4)).batch(batch_size=50, drop_remainder=True)
    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'] - 50, drop_remainder=True)

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
        
        # if args['whitening_flag']:
        if args['augmentation_flag']:
            imageL = augment(imageL, args['trans_range'])
            imageU = augment(imageU, args['trans_range'])
        
        with tf.GradientTape(persistent=True) as tape:
            # supervised
            predL1 = model(imageL)
            predL2 = model(imageL)
            ce_loss = - tf.reduce_mean(tf.reduce_sum(labelL * tf.math.log(tf.clip_by_value(predL1, 1e-10, 1.0)), axis=-1))
            # unsupervised
            predU1 = model(imageU)
            predU2 = model(imageU)
            u_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(predL1 - predL2), axis=-1))
            u_loss += tf.reduce_mean(tf.reduce_sum(tf.math.square(predU1 - predU2), axis=-1))
            
            loss = ce_loss + loss_weight * u_loss
            
        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        
        loss_avg(loss)
        ce_loss_avg(ce_loss)
        u_loss_avg(u_loss)
        probL = tf.nn.softmax(model(imageL, noise=False, training=False), axis=-1)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)

        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'CE_Loss': f'{ce_loss_avg.result():.4f}',
            'U_Loss': f'{u_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
        
    return loss_avg, ce_loss_avg, u_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, split):
    ce_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.Accuracy()

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        pred = model(image, noise=False, training=False)
        ce_loss = - tf.reduce_mean(tf.reduce_sum(label * tf.math.log(tf.clip_by_value(pred, 1e-10, 1.0)), axis=-1))
        pred = tf.stop_gradient(model(image, training=False))
        ce_loss_avg(ce_loss)
        accuracy(tf.argmax(pred, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split}, CE: {ce_loss_avg.result():.4f},  Accuracy: {accuracy.result():.3%}')
    
    return ce_loss_avg, accuracy
#%%
if __name__ == '__main__':
    main()
#%%