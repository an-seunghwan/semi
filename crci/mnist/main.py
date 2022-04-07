#%%
import argparse
import os

os.chdir(r'D:\semi\crci') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/crci') # main directory (repository)

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
from model import VAE
from criterion import ELBO_criterion
from utils import augment, weight_decay_decoupled
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
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--labeled-batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size for labeled dataset (default: 32)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=80, type=int, 
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
    parser.add_argument("--bce_reconstruction", default=True, type=bool,
                        help="Do BCE Reconstruction")

    '''VAE parameters'''
    parser.add_argument('--z_dim', default=6, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--u_dim', default=16, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--classifier_learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate for classifier')
    # parser.add_argument('--weight_decay', default=5e-4, type=float)
    
    parser.add_argument("--z_capacity", default=[0, 7., 100000, 15], type=arg_as_list,
                        help="controlled capacity")
    parser.add_argument("--u_capacity", default=[0, 7., 100000, 15], type=arg_as_list,
                        help="controlled capacity")
    parser.add_argument('--gamma_c', default=15, type=float,
                        help='weight of loss')
    parser.add_argument('--gamma_h', default=30, type=float,
                        help='weight of loss')
    parser.add_argument('--gamma_bc', default=30, type=float,
                        help='weight of loss')
    parser.add_argument('--bc_threshold', default=0.15, type=float,
                        help='threshold of Bhattacharyya coefficient')

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
        plt.imshow(image[i])
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
        plt.imshow(image[i])
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
    
    model = VAE(num_classes=num_classes,
                latent_dim=args['z_dim'], 
                u_dim=args['u_dim'])
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()
    
    # buffer_model = VAE(num_classes=num_classes,
    #             latent_dim=args['z_dim'], 
    #             u_dim=args['u_dim'])
    # buffer_model.build(input_shape=(None, 28, 28, 1))
    # buffer_model.set_weights(model.get_weights()) # weight initialization
    
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['learning_rate'])
    optimizer_classifier = K.optimizers.Adam(learning_rate=args['classifier_learning_rate'])

    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    test_accuracy_print = 0.
    
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        # tf.keras.callbacks.ReduceLROnPlateau
            
        if epoch % args['reconstruct_freq'] == 0:
            loss, recon_loss, elboL_loss, elboU_loss, kl_loss, accuracy, sample_recon = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, num_classes, total_length, test_accuracy_print)
        else:
            loss, recon_loss, elboL_loss, elboU_loss, kl_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, num_classes, total_length, test_accuracy_print)
        # loss, recon_loss, info_loss, nf_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_nf, epoch, args, num_classes, total_length)
        val_recon_loss, val_kl_loss, val_elbo_loss, val_accuracy = validate(val_dataset, model, epoch, beta, args, num_classes, split='Validation')
        test_recon_loss, test_kl_loss, test_elbo_loss, test_accuracy = validate(test_dataset, model, epoch, beta, args, num_classes, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('recon_loss', recon_loss.result(), step=epoch)
            tf.summary.scalar('kl_loss', kl_loss.result(), step=epoch)
            tf.summary.scalar('elboL_loss', elboL_loss.result(), step=epoch)
            tf.summary.scalar('elboU_loss', elboU_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
            if epoch % args['reconstruct_freq'] == 0:
                tf.summary.image("train recon image", sample_recon, step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('recon_loss', val_recon_loss.result(), step=epoch)
            tf.summary.scalar('val_kl_loss', val_kl_loss.result(), step=epoch)
            tf.summary.scalar('elbo_loss', val_elbo_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar('recon_loss', test_recon_loss.result(), step=epoch)
            tf.summary.scalar('test_kl_loss', test_kl_loss.result(), step=epoch)
            tf.summary.scalar('elbo_loss', test_elbo_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            
        test_accuracy_print = test_accuracy.result()

        # Reset metrics every epoch
        loss.reset_states()
        recon_loss.reset_states()
        kl_loss.reset_states()
        elboL_loss.reset_states()
        elboU_loss.reset_states()
        accuracy.reset_states()
        val_recon_loss.reset_states()
        val_kl_loss.reset_states()
        val_elbo_loss.reset_states()
        val_accuracy.reset_states()
        test_recon_loss.reset_states()
        test_kl_loss.reset_states()
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
def train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, num_classes, total_length, test_accuracy_print):
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    elboL_loss_avg = tf.keras.metrics.Mean()
    elboU_loss_avg = tf.keras.metrics.Mean()
    kl_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    autotune = tf.data.AUTOTUNE
    shuffle_and_batchL = lambda dataset: dataset.shuffle(buffer_size=int(1e3)).batch(batch_size=args['labeled_batch_size'], 
                                                                                    drop_remainder=True).prefetch(autotune)
    shuffle_and_batchU = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], 
                                                                                    drop_remainder=True).prefetch(autotune)

    iteratorL = iter(shuffle_and_batchL(datasetL))
    iteratorU = iter(shuffle_and_batchU(datasetU))
        
    iteration = total_length // args['batch_size'] 
    
    # used in computing BC
    BC_valid_mask = np.ones((num_classes, num_classes))
    BC_valid_mask = tf.constant(np.tril(BC_valid_mask, k=-1), tf.float32)
    
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
        
        image = tf.concat([imageL, imageU], axis=0)
        
        # if args['augment']:
        #     imageL_aug = augment(imageL)
        #     imageU_aug = augment(imageU)
        
        '''classifier training (warm-up)'''
        with tf.GradientTape(persistent=True) as tape:    
            prob = model.classify(imageL)
            prob = tf.clip_by_value(prob, 1e-10, 1.)
            ce_loss = tf.reduce_mean(tf.reduce_sum(labelL * tf.math.log(prob), axis=-1))
            
        grads = tape.gradient(ce_loss, model.feature_extractor.trainable_variables + model.h_to_c_logit.trainable_variables) 
        optimizer_classifier.apply_gradients(zip(grads, model.feature_extractor.trainable_variables + model.h_to_c_logit.trainable_variables)) 
            
        '''objective training'''
        with tf.GradientTape(persistent=True) as tape:    
            z_mean, z_logvar, z, c_logit, u_mean, u_logvar, u, xhat = model(image)
            
            '''reconstruction'''
            if args['bce_reconstruction']:
                recon_loss = - tf.reduce_mean(tf.reduce_sum(image * tf.math.log(tf.clip_by_value(xhat, 1e-10, 1.)) + 
                                                            (1. - image) * tf.math.log(1. - tf.clip_by_value(xhat, 1e-10, 1.)), axis=[1, 2, 3]))
            else:
                recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(image - xhat), axis=[1, 2, 3]))
                    
            '''KL-divergence of z'''
            cap_min, cap_max, num_iters, gamma = args['z_capacity']
            num_steps = epoch * iteration + batch_num
            cap_current = (cap_max - cap_min) * (num_steps / num_iters) + cap_min
            cap_current = tf.math.minimum(cap_current, cap_max)
            
            z_kl = tf.reduce_mean(tf.reduce_sum(- 0.5 * (1 + z_logvar - tf.math.pow(z_mean, 2) - tf.math.exp(z_logvar)), axis=-1))
            z_loss = gamma * tf.math.abs(z_kl - cap_current)
            
            '''KL-divergence of c (marginal)'''
            # log_qc = tf.math.reduce_logsumexp(c_logit, axis=0) - tf.math.log(tf.cast(tf.shape(image)[0], tf.float32))
            qc_x = tf.nn.softmax(c_logit, axis=-1)
            qc = tf.reduce_mean(qc_x, axis=0)
            agg_c_kl = tf.reduce_sum(qc * (tf.math.log(tf.clip_by_value(qc, 1e-10, 1.)) - tf.math.log(1. / num_classes)))
            c_loss = args['gamma_c'] * agg_c_kl
            
            '''entropy of c'''
            c_entropy = tf.reduce_mean(- tf.reduce_sum(qc_x * tf.math.log(tf.clip_by_value(qc_x, 1e-10, 1.)), axis=-1))
            c_entropy_loss = args['gamma_h'] * c_entropy
            
            '''mixture KL-divergence of u'''
            cap_min, cap_max, num_iters, gamma = args['u_capacity']
            num_steps = epoch * iteration + batch_num
            cap_current = (cap_max - cap_min) * (num_steps / num_iters) + cap_min
            cap_current = tf.math.minimum(cap_current, cap_max)
            
            u_means = tf.tile(u_mean[..., tf.newaxis], (1, 1, num_classes))
            u_logvars = tf.tile(u_logvar[..., tf.newaxis], (1, 1, num_classes))
            u_kl = tf.reduce_sum(0.5 * (tf.math.pow(u_means - model.u_prior_means, 2) / tf.math.exp(model.u_prior_logvars)
                                        - 1
                                        + tf.math.exp(u_logvars) / tf.math.exp(model.u_prior_logvars)
                                        + model.u_prior_logvars
                                        - u_logvars), axis=-1)
            u_kl = tf.reduce_mean(tf.reduce_sum(tf.multiply(qc_x, u_kl), axis=-1))
            u_loss = gamma * tf.math.abs(u_kl - cap_current)
            
            '''Bhattacharyya coefficient'''
            u_var = tf.math.exp(model.u_prior_logvars)
            avg_u_var = 0.5 * (u_var[tf.newaxis, ...] + u_var[:, tf.newaxis, :])
            inv_avg_u_var = 1. / (avg_u_var + 1e-8)
            diff_mean = model.u_prior_means[tf.newaxis, ...] + model.u_prior_means[:, tf.newaxis, :]
            D = 1/8 * tf.reduce_sum(diff_mean * inv_avg_u_var * diff_mean, axis=-1)
            D += 0.5 * tf.reduce_sum(tf.math.log(avg_u_var + 1e-8), axis=-1)
            D += - 0.25 * tf.reduce_sum(model.u_prior_logvars, axis=-1)[tf.newaxis, ...] + tf.reduce_sum(model.u_prior_logvars, axis=-1)[:, tf.newaxis]
            BC = tf.math.exp(D)
            valid_BC = BC * BC_valid_mask
            prior_intersection_loss = args['gamma_bc'] * tf.reduce_sum(tf.math.maximum(valid_BC - args['bc_threshold'], 0))
            
            loss = recon_loss + z_loss + c_loss + c_entropy_loss + u_loss + prior_intersection_loss
            
        grads = tape.gradient(loss, model) 
        optimizer.apply_gradients(zip(grads, model))
        # '''decoupled weight decay'''
        # weight_decay_decoupled(model, buffer_model, decay_rate=args['weight_decay'] * optimizer.lr)
        
        loss_avg(loss)
        elboL_loss_avg(elboL)
        elboU_loss_avg(elboU)
        recon_loss_avg(recon_loss)
        kl_loss_avg(qz - pz)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)
        
        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'ELBO(L)': f'{elboL_loss_avg.result():.4f}',
            'ELBO(U)': f'{elboU_loss_avg.result():.4f}',
            'Recon': f'{recon_loss_avg.result():.4f}',
            'KL': f'{kl_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}',
            'beta': f'{beta:.4f}'
        })
    
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images1(model, xhat)
        generate_and_save_images2(model, xhat, epoch, f'logs/{args["dataset"]}_{args["labeled_examples"]}/{current_time}')
        return loss_avg, recon_loss_avg, elboL_loss_avg, elboU_loss_avg, kl_loss_avg, accuracy, sample_recon
    else:
        return loss_avg, recon_loss_avg, elboL_loss_avg, elboU_loss_avg, kl_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, beta, args, num_classes, split):
    recon_loss_avg = tf.keras.metrics.Mean()   
    kl_loss_avg = tf.keras.metrics.Mean()   
    elbo_loss_avg = tf.keras.metrics.Mean()   
    accuracy = tf.keras.metrics.Accuracy()
    
    dataset = dataset.batch(args['batch_size'], drop_remainder=False)
    for image, label in dataset:
        mean, logvar, z, xhat = model([image, label], training=False)
        recon_loss, prior_y, pz, qz = ELBO_criterion(xhat, image, label, z, mean, logvar, num_classes, args)
        elbo = tf.reduce_mean(recon_loss - prior_y + beta * (qz - pz))
        prob = model.classify(image, training=False)
        
        recon_loss_avg(recon_loss)
        kl_loss_avg(qz - pz)
        elbo_loss_avg(elbo)
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} ELBO Loss: {elbo_loss_avg.result():.4f}, Recon: {recon_loss_avg.result():.4f}, KL: {kl_loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    
    return recon_loss_avg, kl_loss_avg, elbo_loss_avg, accuracy
#%%
# def weight_schedule(epoch, epochs, weight_max):
#     return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
if __name__ == '__main__':
    main()
#%%