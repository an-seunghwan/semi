#%%
import argparse
import os

# os.chdir(r'D:\EXoN_official') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/EXoN_official') # main directory (repository)
os.chdir('/Users/anseunghwan/Documents/GitHub/semi/adgm')

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
from model import ADGM
from criterion import ELBO_criterion
from utils import weight_decay_decoupled
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

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset used for training')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--labeled-batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size of labeled dataset')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=30, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=100, 
                        help='number labeled examples (default: 100), all labels are balanced')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters'''
    parser.add_argument('--bce', "--bce_reconstruction", default=True, type=bool,
                        help="Do BCE Reconstruction")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=32, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--aux_dim', "--aux_dim_continuous", default=32, type=int,
                        metavar='Auxiliary Dim For Continuous Variable',
                        help='feature dimension in auxiliary latent space for continuous variable')
    parser.add_argument('--beta', default=0.1, type=float,
                        metavar='beta',
                        help='weight of KL-divergence')
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning_rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', '--weight_decay', default=1, type=float)
    parser.add_argument('--clipvalue', default=1, type=float)
    # parser.add_argument('--clipnorm', default=5, type=float)

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
def generate_and_save_images1(model, image):
    image = image.numpy()
    
    buf = io.BytesIO()
    figure = plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(image[i][..., 0])
        plt.axis('off')
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=1)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def generate_and_save_images2(model, image, step, save_dir):
    image = image.numpy()
    
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(image[i][..., 0])
        plt.axis('off')
    plt.savefig('{}/image_at_epoch_{}.png'.format(save_dir, step))
    # plt.show()
    plt.close()
#%%
def main():
    '''argparse to dictionary'''
    args = vars(get_args())
    # '''argparse debugging'''
    # args = vars(parser.parse_args(args=['--config_path', 'configs/mnist_100.yaml']))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    total_length = sum(1 for _ in datasetU)
    
    model = ADGM(args,
                num_classes,
                a_dim=args['aux_dim'],
                latent_dim=args['latent_dim'])
    model.classifier.build(input_shape=[(None, 784), (None, args['aux_dim'])])
    model.build(input_shape=[(None, 28, 28, 1), (None, num_classes)])
    model.summary()
    
    buffer_model = ADGM(args,
                num_classes,
                a_dim=args['aux_dim'],
                latent_dim=args['latent_dim'])
    buffer_model.classifier.build(input_shape=[(None, 784), (None, args['aux_dim'])])
    buffer_model.build(input_shape=[(None, 28, 28, 1), (None, num_classes)])
    buffer_model.set_weights(model.get_weights()) # weight initialization
    
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['lr'],
                                  clipvalue=args['clipvalue'])

    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    test_accuracy_print = 0.
    
    # initial beta
    beta = tf.cast(args['beta'], tf.float32)
    
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        lr_gamma = 0.8
        min_lr = args['lr'] * 0.1
        if optimizer.lr * lr_gamma < min_lr:
            optimizer.lr = min_lr
        elif epoch % 35 == 0:
            optimizer.lr = optimizer.lr * lr_gamma
            
        if epoch % args['reconstruct_freq'] == 0:
            loss, recon_loss, elboL_loss, elboU_loss, kl_loss, kl_aux_loss, accuracy, sample_recon = train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, beta, num_classes, total_length, test_accuracy_print)
        else:
            loss, recon_loss, elboL_loss, elboU_loss, kl_loss, kl_aux_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, beta, num_classes, total_length, test_accuracy_print)
        # loss, recon_loss, info_loss, nf_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_nf, epoch, args, num_classes, total_length)
        val_recon_loss, val_kl_loss, val_kl_aux_loss, val_elbo_loss, val_accuracy = validate(val_dataset, model, epoch, beta, args, num_classes, split='Validation')
        test_recon_loss, test_kl_loss, test_kl_aux_loss, test_elbo_loss, test_accuracy = validate(test_dataset, model, epoch, beta, args, num_classes, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('recon_loss', recon_loss.result(), step=epoch)
            tf.summary.scalar('kl_loss', kl_loss.result(), step=epoch)
            tf.summary.scalar('kl_aux_loss', kl_aux_loss.result(), step=epoch)
            tf.summary.scalar('elboL_loss', elboL_loss.result(), step=epoch)
            tf.summary.scalar('elboU_loss', elboU_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
            if epoch % args['reconstruct_freq'] == 0:
                tf.summary.image("train recon image", sample_recon, step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('recon_loss', val_recon_loss.result(), step=epoch)
            tf.summary.scalar('val_kl_loss', val_kl_loss.result(), step=epoch)
            tf.summary.scalar('val_kl_aux_loss', val_kl_aux_loss.result(), step=epoch)
            tf.summary.scalar('elbo_loss', val_elbo_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar('recon_loss', test_recon_loss.result(), step=epoch)
            tf.summary.scalar('test_kl_loss', test_kl_loss.result(), step=epoch)
            tf.summary.scalar('test_kl_aux_loss', test_kl_loss.result(), step=epoch)
            tf.summary.scalar('elbo_loss', test_elbo_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            
        test_accuracy_print = test_accuracy.result()

        # Reset metrics every epoch
        loss.reset_states()
        recon_loss.reset_states()
        kl_loss.reset_states()
        kl_aux_loss.reset_states()
        elboL_loss.reset_states()
        elboU_loss.reset_states()
        accuracy.reset_states()
        val_recon_loss.reset_states()
        val_kl_loss.reset_states()
        val_kl_aux_loss.reset_states()
        val_elbo_loss.reset_states()
        val_accuracy.reset_states()
        test_recon_loss.reset_states()
        test_kl_loss.reset_states()
        test_kl_aux_loss.reset_states()
        test_elbo_loss.reset_states()
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
def train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, beta, num_classes, total_length, test_accuracy_print):
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    elboL_loss_avg = tf.keras.metrics.Mean()
    elboU_loss_avg = tf.keras.metrics.Mean()
    kl_loss_avg = tf.keras.metrics.Mean()
    kl_aux_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    '''supervised classification weight'''
    # alpha = tf.cast(0.1 * (total_length + args['labeled_examples']) / args['labeled_examples'], tf.float32)
    alpha = tf.cast((total_length + args['labeled_examples']) / args['labeled_examples'], tf.float32)
    
    autotune = tf.data.AUTOTUNE
    batchL = lambda dataset: dataset.shuffle(buffer_size=int(1e3)).batch(batch_size=args['labeled_batch_size'], drop_remainder=False).prefetch(autotune)
    shuffle_and_batchU = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True).prefetch(autotune)

    iteratorL = iter(batchL(datasetL))
    iteratorU = iter(shuffle_and_batchU(datasetU))
        
    iteration = total_length // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(batchL(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, _ = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batchU(datasetU))
            imageU, _ = next(iteratorU)
        
        # if args['augment']:
        #     imageL_aug = augment(imageL)
        #     imageU_aug = augment(imageU)
            
        with tf.GradientTape(persistent=True) as tape:    
            '''labeled'''
            mean, logvar, z, xhat, a, qa_mean, qa_logvar, pa_mean, pa_logvar = model([imageL, labelL])
            recon_loss, prior_y, pz, qz, pa, qa = ELBO_criterion(xhat, imageL, labelL, z, mean, logvar, num_classes, args,
                                                                a, qa_mean, qa_logvar, pa_mean, pa_logvar)
            elboL = tf.reduce_mean(recon_loss - prior_y + beta * (qz - pz) + beta * (qa - pa))
            
            '''unlabeled'''
            with tape.stop_recording():
                labelU = tf.concat([tf.one_hot(i, depth=num_classes)[tf.newaxis, ] for i in range(num_classes)], axis=0)[tf.newaxis, ...]
                labelU = tf.repeat(labelU, tf.shape(imageU)[0], axis=0)
                labelU = tf.reshape(labelU, (-1, num_classes))
                
            imageU_ = imageU[:, tf.newaxis, :, :, :]
            imageU_ = tf.reshape(tf.repeat(imageU_, num_classes, axis=1), (-1, 28, 28, 1))
            
            mean, logvar, z, xhat, a, qa_mean, qa_logvar, pa_mean, pa_logvar = model([imageU_, labelU])
            recon_loss, prior_y, pz, qz, pa, qa = ELBO_criterion(xhat, imageU_, labelU, z, mean, logvar, num_classes, args,
                                                                 a, qa_mean, qa_logvar, pa_mean, pa_logvar)
            
            recon_loss = tf.reshape(recon_loss, (tf.shape(imageU)[0], num_classes, -1))
            prior_y = tf.reshape(prior_y, (tf.shape(imageU)[0], num_classes, -1))
            pz = tf.reshape(pz, (tf.shape(imageU)[0], num_classes, -1))
            qz = tf.reshape(qz, (tf.shape(imageU)[0], num_classes, -1))
            pa = tf.reshape(pa, (tf.shape(imageU)[0], num_classes, -1))
            qa = tf.reshape(qa, (tf.shape(imageU)[0], num_classes, -1))
            
            probU = model.classify(imageU)
            elboU = recon_loss - prior_y + beta * (qz - pz) + beta * (qa - pa)
            elboU = tf.reduce_mean(tf.reduce_sum(probU[..., tf.newaxis] * elboU, axis=[1, 2]))
            entropyU = - tf.reduce_mean(tf.reduce_sum(probU * tf.math.log(tf.clip_by_value(probU, 1e-8, 1.)), axis=-1))
            elboU -= entropyU
            
            '''supervised classification loss'''
            probL = model.classify(imageL)
            cce = - tf.reduce_mean(tf.reduce_sum(tf.multiply(labelL, tf.math.log(tf.clip_by_value(probL, 1e-10, 1.))), axis=-1))
            
            loss = elboL + elboU + alpha * cce
            
        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model, buffer_model, decay_rate=args['wd'] * optimizer.lr)
        
        loss_avg(loss)
        elboL_loss_avg(elboL)
        elboU_loss_avg(elboU)
        recon_loss_avg(recon_loss)
        kl_loss_avg(qz - pz)
        kl_aux_loss_avg(qa - pa)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)
        
        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'ELBO(L)': f'{elboL_loss_avg.result():.4f}',
            'ELBO(U)': f'{elboU_loss_avg.result():.4f}',
            'Recon': f'{recon_loss_avg.result():.4f}',
            'KL': f'{kl_loss_avg.result():.4f}',
            'KL_aux': f'{kl_aux_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}',
            'beta': f'{beta:.4f}'
        })
    
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images1(model, xhat)
        generate_and_save_images2(model, xhat, epoch, f'logs/{args["dataset"]}_{args["labeled_examples"]}/{current_time}')
        return loss_avg, recon_loss_avg, elboL_loss_avg, elboU_loss_avg, kl_loss_avg, kl_aux_loss_avg, accuracy, sample_recon
    else:
        return loss_avg, recon_loss_avg, elboL_loss_avg, elboU_loss_avg, kl_loss_avg, kl_aux_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, beta, args, num_classes, split):
    recon_loss_avg = tf.keras.metrics.Mean()   
    kl_loss_avg = tf.keras.metrics.Mean()   
    kl_aux_loss_avg = tf.keras.metrics.Mean()   
    elbo_loss_avg = tf.keras.metrics.Mean()   
    accuracy = tf.keras.metrics.Accuracy()
    
    dataset = dataset.batch(args['batch_size'], drop_remainder=False)
    for image, label in dataset:
        mean, logvar, z, xhat, a, qa_mean, qa_logvar, pa_mean, pa_logvar = model([image, label], training=False)
        recon_loss, prior_y, pz, qz, pa, qa = ELBO_criterion(xhat, image, label, z, mean, logvar, num_classes, args,
                                                            a, qa_mean, qa_logvar, pa_mean, pa_logvar)
        elbo = tf.reduce_mean(recon_loss - prior_y + beta * (qz - pz) + beta * (qa - pa))
        prob = model.classify(image, training=False)
        
        recon_loss_avg(recon_loss)
        kl_loss_avg(qz - pz)
        kl_aux_loss_avg(qa - pa)
        elbo_loss_avg(elbo)
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} ELBO Loss: {elbo_loss_avg.result():.4f}, Recon: {recon_loss_avg.result():.4f}, KL: {kl_loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    
    return recon_loss_avg, kl_loss_avg, kl_aux_loss_avg, elbo_loss_avg, accuracy
#%%
# def weight_schedule(epoch, epochs, weight_max):
#     return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
if __name__ == '__main__':
    main()
#%%