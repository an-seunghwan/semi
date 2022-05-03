#%%
import argparse
import os

os.chdir(r'D:\semi\lp') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/lp') # main directory (repository)

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
from model_large import CNN
from utils import weight_decay_decoupled, linear_rampup, cosine_rampdown, augment, build_pseudo_label
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
    
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results (ex. generating color MNIST)')
    parser.add_argument('--epochs', default=180, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-label_b', '--labeled-batch-size', default=50, type=int,
                        metavar='N', help='labeled examples per minibatch (default: 256)')
    
    parser.add_argument('--learning-rate', default=0.01, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=210, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--weight-decay', default=2e-4, type=float,
                        metavar='W', help='weight decay (default: 2e-4)')
    
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    parser.add_argument('--dfs-k', type=int, default=50,
                        help='diffusion k')
    parser.add_argument('--isL2', default=True, type=bool,
                        help='is l2 normalized features')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled instances')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    
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

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}_{args["num_labeled"]}'

    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    total_length = sum(1 for _ in datasetU)
    
    '''load model from stage1'''
    model_path = log_path + '/stage1'
    model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
    model = CNN(num_classes, args['isL2'])
    model.build(input_shape=(None, 32, 32, 3))
    model.load_weights(model_path + '/' + model_name)
    model.summary()
    
    buffer_model = CNN(num_classes, args['isL2'])
    buffer_model.build(input_shape=(None, 32, 32, 3))
    buffer_model.set_weights(model.get_weights()) # weight initialization
    
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['learning_rate'])
    
    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    for epoch in range(args['start_epoch'], args['epochs']):
        
        loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, num_classes, total_length)
        val_loss, val_accuracy = validate(val_dataset, model, epoch, args, split='Validation')
        test_loss, test_accuracy = validate(test_dataset, model, epoch, args, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        # Reset metrics every epoch
        loss.reset_states()
        accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        test_loss.reset_states()
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
def train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, num_classes, total_length):
    loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    autotune = tf.data.AUTOTUNE
    shuffle_and_batchL = lambda dataset: dataset.shuffle(buffer_size=int(1e5)).batch(batch_size=args['labeled_batch_size'], drop_remainder=False).prefetch(autotune)
    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'] - args['labeled_batch_size'], drop_remainder=True).prefetch(autotune)
    
    pseudo_datasetU, class_weights, accL = build_pseudo_label(datasetL, datasetU, model, num_classes, args, k=args['dfs_k'])
    class_weights = tf.cast(class_weights, tf.float32)
    
    iteratorL = iter(shuffle_and_batchL(datasetL))
    iteratorU = iter(shuffle_and_batch(pseudo_datasetU))
        
    iteration = (50000 - args['num_labeled']) // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        '''learning rate schedule'''
        epoch_ = epoch + batch_num / iteration
        lr = linear_rampup(epoch_, args['lr_rampup']) * (optimizer.lr - args['initial_lr']) + args['initial_lr']
        if args['lr_rampdown_epochs']:
            lr *= cosine_rampdown(epoch_, args['lr_rampdown_epochs'])
        optimizer.lr = lr
        
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(shuffle_and_batchL(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, labelU, weightU = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(pseudo_datasetU))
            imageU, labelU, weightU = next(iteratorU)
        
        '''augmentation'''
        imageL = augment(imageL)
        imageU = augment(imageU)
        
        '''concat'''
        image = tf.concat([imageL, imageU], axis=0)
        label = tf.concat([labelL, labelU], axis=0)
        weights = tf.concat([tf.ones((args['labeled_batch_size'], )), weightU], axis=0)
        
        '''normalization'''
        channel_stats = dict(mean=tf.reshape(tf.cast(np.array([0.4914, 0.4822, 0.4465]), tf.float32), (1, 1, 1, 3)),
                             std=tf.reshape(tf.cast(np.array([0.2470, 0.2435, 0.2616]), tf.float32), (1, 1, 1, 3)))
        image -= channel_stats['mean']
        image /= channel_stats['std']
        
        with tf.GradientTape(persistent=True) as tape:
            '''both labeled and unlabeled'''
            pred, _ = model(image)
            pred = tf.nn.softmax(pred, axis=-1)
            ce_loss = tf.reduce_sum(class_weights * label * tf.math.log(tf.clip_by_value(pred, 1e-10, 1.0)), axis=-1)
            loss = - tf.reduce_sum(weights * ce_loss) / args['batch_size']
            
        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model, buffer_model, decay_rate=args['weight_decay'] * optimizer.lr)
        
        loss_avg(loss)
        imageL -= channel_stats['mean']
        imageL /= channel_stats['std']
        probL, _ = model(imageL, training=False)
        probL = tf.nn.softmax(probL, axis=-1)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)

        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'LP Accuracy': f'{accL:.3%}'
        })
        
    return loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, split):
    loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.Accuracy()

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        '''normalization'''
        channel_stats = dict(mean=tf.reshape(tf.cast(np.array([0.4914, 0.4822, 0.4465]), tf.float32), (1, 1, 1, 3)),
                            std=tf.reshape(tf.cast(np.array([0.2470, 0.2435, 0.2616]), tf.float32), (1, 1, 1, 3)))
        image -= channel_stats['mean']
        image /= channel_stats['std']
        
        pred, _ = model(image, training=False)
        pred = tf.nn.softmax(pred, axis=-1)
        ce_loss = - tf.reduce_mean(tf.reduce_sum(label * tf.math.log(tf.clip_by_value(pred, 1e-10, 1.0)), axis=-1))
        loss_avg(ce_loss)
        accuracy(tf.argmax(pred, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split}, Loss: {loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    
    return loss_avg, accuracy
#%%
if __name__ == '__main__':
    main()
#%%