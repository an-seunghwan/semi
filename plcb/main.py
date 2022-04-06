#%%
import argparse
import os

# os.chdir(r'D:\semi\plcb') # main directory (repository)
os.chdir('/home1/seunghwan/semi/plcb') # main directory (repository)

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
from utils import augment, non_smooth_mixup, weight_decay_decoupled
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
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results (ex. generating color MNIST)')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='mini-batch size (default: 100)')
    parser.add_argument('--labeled_batch_size', default=16, type=int, metavar='N', 
                        help="Labeled examples per minibatch (default: no constrain)")

    '''SSL Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=400, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='CNN dropout')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay')
    parser.add_argument('--reg1', type=float, default=0.8, 
                        help='Hyperparam for loss')
    parser.add_argument('--reg2', type=float, default=0.4, 
                        help='Hyperparam for loss')
    parser.add_argument("--M", default=[250, 350], type=arg_as_list,
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    # parser.add_argument('--swa', type=str, default='True', help='Apply SWA')
    # parser.add_argument('--swa_start', type=int, default=350, help='Start SWA')
    # parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')
    # parser.add_argument('--swa_lr', type=float, default=-0.01, help='LR')
    
    parser.add_argument('--Mixup_Alpha', type=float, default=1, 
                        help='Alpha value for the beta dist from mixup')
    parser.add_argument('--DApseudolab', type=bool, default=False, 
                        help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--drop_extra_forward', type=bool, default=True, 
                        help='Do an extra forward pass to compute the labels without dropout.')

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
    # args = vars(parser.parse_args(args=['--config_path', 'configs/cifar10_4000.yaml']))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    total_length = sum(1 for _ in datasetU)
    
    model_path = log_path + '/warmup'
    model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
    model = CNN(num_classes, 
                dropratio=args['dropout'])
    model.build(input_shape=(None, 32, 32, 3))
    model.load_weights(model_path + '/' + model_name)
    model.summary()
    
    buffer_model = CNN(num_classes, 
                    dropratio=args['dropout'])
    buffer_model.build(input_shape=(None, 32, 32, 3))
    buffer_model.set_weights(model.get_weights()) # weight initialization
    
    if args['drop_extra_forward']:
        pslab_model = CNN(num_classes, 
                        dropratio=0.0) # without dropout
    else:
        pslab_model = CNN(num_classes, 
                        dropratio=args['dropout']) 
    pslab_model.build(input_shape=(None, 32, 32, 3))
    pslab_model.set_weights(model.get_weights()) # weight initialization
    
    '''optimizer'''
    # optimizer = K.optimizers.SGD(learning_rate=args['learning_rate'],
    #                             momentum=args['momentum'])
    optimizer = K.optimizers.Adam(learning_rate=args['learning_rate'])
    
    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    for epoch in range(args['start_epoch'], args['epochs']):
        
        for ad_num, ad_epoch in enumerate(args['M'] + [args['epochs']]):
            if epoch < ad_epoch:
                optimizer.lr = args['learning_rate'] * (args['lr_gamma'] ** ad_num)
                break
        
        loss, mixup_loss, rega_loss, regb_loss, accuracy = train(datasetL, datasetU, model, buffer_model, pslab_model, optimizer, epoch, args, num_classes, total_length)
        val_loss, val_ce_loss, val_rega_loss, val_regb_loss, val_accuracy = validate(val_dataset, model, epoch, args, num_classes, split='Validation')
        test_loss, test_ce_loss, test_rega_loss, test_regb_loss, test_accuracy = validate(test_dataset, model, epoch, args, num_classes, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('mixup_loss', mixup_loss.result(), step=epoch)
            tf.summary.scalar('rega_loss', rega_loss.result(), step=epoch)
            tf.summary.scalar('regb_loss', regb_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('ce_loss', val_ce_loss.result(), step=epoch)
            tf.summary.scalar('rega_loss', val_rega_loss.result(), step=epoch)
            tf.summary.scalar('regb_loss', val_regb_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('ce_loss', test_ce_loss.result(), step=epoch)
            tf.summary.scalar('rega_loss', test_rega_loss.result(), step=epoch)
            tf.summary.scalar('regb_loss', test_regb_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        # Reset metrics every epoch
        loss.reset_states()
        mixup_loss.reset_states()
        rega_loss.reset_states()
        regb_loss.reset_states()
        accuracy.reset_states()
        val_loss.reset_states()
        val_ce_loss.reset_states()
        val_rega_loss.reset_states()
        val_regb_loss.reset_states()
        val_accuracy.reset_states()
        test_loss.reset_states()
        test_ce_loss.reset_states()
        test_rega_loss.reset_states()
        test_regb_loss.reset_states()
        test_accuracy.reset_states()
        
        '''update pseudo-label model for every epoch'''
        pslab_model.set_weights(model.get_weights())

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
def train(datasetL, datasetU, model, buffer_model, pslab_model, optimizer, epoch, args, num_classes, total_length):
    loss_avg = tf.keras.metrics.Mean()
    mixup_loss_avg = tf.keras.metrics.Mean()
    rega_loss_avg = tf.keras.metrics.Mean()
    regb_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    shuffle_and_batchL = lambda dataset: dataset.shuffle(buffer_size=int(1e4)).batch(batch_size=args['labeled_batch_size'], drop_remainder=True)
    shuffle_and_batchU = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'] - args['labeled_batch_size'], drop_remainder=True)

    iteratorL = iter(shuffle_and_batchL(datasetL))
    iteratorU = iter(shuffle_and_batchU(datasetU))
        
    iteration = total_length // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(shuffle_and_batchL(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, _ = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batchU(datasetU))
            imageU, _ = next(iteratorU)
        
        imageL_aug = augment(imageL)
        imageU_aug = augment(imageU)
            
        '''normalization'''
        channel_stats = dict(mean=tf.reshape(tf.cast(np.array([0.4914, 0.4822, 0.4465]), tf.float32), (1, 1, 1, 3)),
                             std=tf.reshape(tf.cast(np.array([0.2470, 0.2435, 0.2616]), tf.float32), (1, 1, 1, 3)))
        imageL -= channel_stats['mean']
        imageL /= channel_stats['std']
        imageU -= channel_stats['mean']
        imageU /= channel_stats['std']
        
        imageL_aug -= channel_stats['mean']
        imageL_aug /= channel_stats['std']
        imageU_aug -= channel_stats['mean']
        imageU_aug /= channel_stats['std']
        
        mix_weight = tf.constant(np.random.beta(args['Mixup_Alpha'], args['Mixup_Alpha']))
        
        with tf.GradientTape(persistent=True) as tape:
            '''pseudo-label and mix-up'''
            with tape.stop_recording():
                labelU = pslab_model(imageU, training=False) # without augmentation and dropout
                labelU = tf.nn.softmax(labelU, axis=-1)
                image = tf.concat([imageL_aug, imageU_aug], axis=0)
                label = tf.concat([labelL, labelU], axis=0)
                image_mix, label_shuffle = non_smooth_mixup(image, label, mix_weight)
                
            pred = model(image_mix)
            prob = tf.nn.softmax(pred, axis=-1)
            prob = tf.clip_by_value(prob, 1e-10, 1.0) 
            
            mixup_loss = - mix_weight * tf.reduce_mean(tf.reduce_sum(label_shuffle * tf.math.log(prob), axis=-1))
            mixup_loss += - (1. - mix_weight) * tf.reduce_mean(tf.reduce_sum(label * tf.math.log(prob), axis=-1))
            
            prob_avg = tf.reduce_mean(prob, axis=0)
            prob_avg = tf.clip_by_value(prob_avg, 1e-10, 1.0)
            # RegA = tf.reduce_sum(1./num_classes * (tf.math.log(tf.clip_by_value(1./num_classes, 1e-10, 1.0)) - tf.math.log(tf.clip_by_value(prob_avg, 1e-10, 1.0))))
            RegA = - tf.reduce_sum(1./num_classes * tf.math.log(prob_avg))
            RegB = - tf.reduce_mean(tf.reduce_sum(prob * tf.math.log(prob), axis=-1))
            
            loss = mixup_loss + args['reg1'] * RegA + args['reg2'] * RegB
            
        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model, buffer_model, decay_rate=args['weight_decay'] * optimizer.lr)
        
        loss_avg(loss)
        mixup_loss_avg(mixup_loss)
        rega_loss_avg(RegA)
        regb_loss_avg(RegB)
        probL = model(imageL, training=False)
        probL = tf.nn.softmax(probL, axis=-1)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)

        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'MIXUP_Loss': f'{mixup_loss_avg.result():.4f}',
            'RegA_Loss': f'{rega_loss_avg.result():.4f}',
            'RegB_Loss': f'{regb_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
        
    return loss_avg, mixup_loss_avg, rega_loss_avg, regb_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, num_classes, split):
    loss_avg = tf.keras.metrics.Mean()
    ce_loss_avg = tf.keras.metrics.Mean()
    rega_loss_avg = tf.keras.metrics.Mean()
    regb_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.Accuracy()

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        channel_stats = dict(mean=tf.reshape(tf.cast(np.array([0.4914, 0.4822, 0.4465]), tf.float32), (1, 1, 1, 3)),
                             std=tf.reshape(tf.cast(np.array([0.2470, 0.2435, 0.2616]), tf.float32), (1, 1, 1, 3)))
        image -= channel_stats['mean']
        image /= channel_stats['std']
        
        pred = model(image, training=False)
        prob = tf.nn.softmax(pred, axis=-1)
        prob = tf.clip_by_value(prob, 1e-10, 1.0)
        
        ce_loss = - tf.reduce_mean(tf.reduce_sum(label * tf.math.log(prob), axis=-1))
        prob_avg = tf.reduce_mean(prob, axis=0)
        prob_avg = tf.clip_by_value(prob_avg, 1e-10, 1.0)
        # RegA = tf.reduce_sum(1./num_classes * (tf.math.log(tf.clip_by_value(1./num_classes, 1e-10, 1.0)) - tf.math.log(tf.clip_by_value(prob_avg, 1e-10, 1.0))))
        RegA = - tf.reduce_sum(1./num_classes * tf.math.log(prob_avg))
        RegB = - tf.reduce_mean(tf.reduce_sum(prob * tf.math.log(prob), axis=-1))
        
        loss = ce_loss + args['reg1'] * RegA + args['reg2'] * RegB
        
        loss_avg(loss)
        ce_loss_avg(ce_loss)
        rega_loss_avg(RegA)
        regb_loss_avg(RegB)
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split}, Loss: {loss_avg.result():.4f},  CE: {ce_loss_avg.result():.4f},  RegA: {rega_loss_avg.result():.4f},  RegB: {regb_loss_avg.result():.4f},  Accuracy: {accuracy.result():.3%}')
    
    return loss_avg, ce_loss_avg, rega_loss_avg, regb_loss_avg, accuracy
#%%
if __name__ == '__main__':
    main()
#%%