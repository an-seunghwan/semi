#%%
'''
220111: mutual information is not included yet
'''
#%%
import argparse
import os

os.chdir(r'D:\semi\semi\unconditional') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/semi/unconditional') # main directory (repository)

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
from mixup import augment, label_smoothing
#%%
# config = tf.compat.v1.ConfigProto()
# '''
# GPU 메모리를 전부 할당하지 않고, 아주 적은 비율만 할당되어 시작됨
# 프로세스의 메모리 수요에 따라 자동적으로 증가
# but
# GPU 메모리를 처음부터 전체 비율을 사용하지 않음
# '''
# config.gpu_options.allow_growth = True

# '''
# 분산 학습 설정
# '''
# strategy = tf.distribute.MirroredStrategy()
# session = tf.compat.v1.InteractiveSession(config=config)
#%%
import ast
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
#%%
# def get_args():
parser = argparse.ArgumentParser('parameters')

# parser.add_argument('-bp', '--base_path', default=".")
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra, cmnist)')
# parser.add_argument('-is', "--image-size", default=[32, 32], type=arg_as_list,
#                     metavar='Image Size List', help='the size of h * w for image')
# parser.add_argument("--channel", default=3, type=int,
#                     metavar='Channel size', help='the size of image channel')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')

'''SSL VAE Train PreProcess Parameter'''
parser.add_argument('--epochs', default=600, type=int, 
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, 
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--reconstruct-freq', '-rf', default=50, type=int,
                    metavar='N', help='reconstruct frequency (default: 50)')
parser.add_argument('--labeled_examples', type=int, default=4000, 
                    help='number labeled examples (default: 4000')
parser.add_argument('--validation_examples', type=int, default=5000, 
                    help='number validation examples (default: 5000')
parser.add_argument('--augment', action='store_true', 
                    help="apply augmentation to image")

'''Deep VAE Model Parameters'''
# parser.add_argument('--net-name', default="wideresnet-28-2", type=str, help="the name for network to use")
parser.add_argument('--depth', type=int, default=28, 
                    help='depth for WideResnet (default: 28)')
parser.add_argument('--width', type=int, default=2, 
                    help='widen factor for WideResnet (default: 2)')
parser.add_argument('--slope', type=float, default=0.1, 
                    help='slope parameter for LeakyReLU (default: 0.1)')
parser.add_argument('-dr', '--drop-rate', default=0, type=float, 
                    help='drop rate for the network')
parser.add_argument("--br", "--bce-reconstruction", action='store_true', 
                    help='Do BCE Reconstruction')
parser.add_argument("-s", "--x-sigma", default=1, type=float,
                    help="The standard variance for reconstructed images, work as regularization")

'''VAE parameters'''
parser.add_argument('--latent_dim', "--latent_dim_continuous", default=128, type=int,
                    metavar='Latent Dim For Continuous Variable',
                    help='feature dimension in latent space for continuous variable')

'''VAE Loss Function Parameters'''
# parser.add_argument("-ei", "--evaluate-inference", action='store_true',
#                     help='Calculate the inference accuracy for unlabeled dataset')
parser.add_argument('--kbmc', '--kl-beta-max-continuous', default=1e-3, type=float, 
                    metavar='KL Beta', help='the epoch to linear adjust kl beta')
parser.add_argument('--kbmd', '--kl-beta-max-discrete', default=1e-3, type=float, 
                    metavar='KL Beta', help='the epoch to linear adjust kl beta')
parser.add_argument('--akb', '--adjust-kl-beta-epoch', default=200, type=int, 
                    metavar='KL Beta', help='the max epoch to adjust kl beta')
parser.add_argument('--ewm', '--elbo-weight-max', default=1e-3, type=float, 
                    metavar='weight for elbo loss part')
parser.add_argument('--aew', '--adjust-elbo-weight', default=400, type=int,
                    metavar="the epoch to adjust elbo weight to max")
parser.add_argument('--wrd', default=1, type=float,
                    help="the max weight for the optimal transport estimation of discrete variable c")
parser.add_argument('--wmf', '--weight-modify-factor', default=0.4, type=float,
                    help="weight  will get wrz at amf * epochs")
parser.add_argument('--pwm', '--posterior-weight-max', default=1, type=float,
                    help="the max value for posterior weight")
parser.add_argument('--apw', '--adjust-posterior-weight', default=200, type=float,
                    help="adjust posterior weight")

'''Optimizer Parameters'''
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b1', '--beta1', default=0.9, type=float, metavar='Beta1 In ADAM and SGD',
                    help='beta1 for adam as well as momentum for SGD')
# parser.add_argument('-ad', "--adjust-lr", default=[400, 500, 550], type=arg_as_list,
#                     help="The milestone list for adjust learning rate")
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)

'''Normalizing Flow Model Parameters'''
parser.add_argument('--z_mask', default='checkerboard', type=str,
                    help='mask type of continuous latent for Real NVP (e.g. checkerboard or half)')
parser.add_argument('--c_mask', default='half', type=str,
                    help='mask type of discrete latent for Real NVP (e.g. checkerboard or half)')
parser.add_argument('--z_emb', default=256, type=int,
                    help='embedding dimension of continuous latent for coupling layer')
parser.add_argument('--c_emb', default=256, type=int,
                    help='embedding dimension of discrete latent for coupling layer')
parser.add_argument('--K1', default=8, type=int,
                    help='number of coupling layers in Real NVP (continous latent)')
parser.add_argument('--K2', default=8, type=int,
                    help='number of coupling layers in Real NVP (discrete latent)')
parser.add_argument('--coupling_MLP_num', default=4, type=int,
                    help='number of dense layers in single coupling layer')

'''Normalizing Flow Optimizer Parameters'''
parser.add_argument('--lr_nf', '--learning-rate-nf', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for normalizing flow')
parser.add_argument('--reg', default=0.01, type=float,
                    help='L2 regularization parameter for dense layers in Real NVP')
parser.add_argument('--decay_steps', default=1, type=int,
                    help='decay steps for exponential decay schedule')
parser.add_argument('--decay_rate', default=0.95, type=float,
                    help='decay rate for exponential decay schedule')
parser.add_argument('--gradclip', default=1., type=float,
                    help='gradclip value')

'''Optimizer Transport Estimation Parameters'''
parser.add_argument('--epsilon', default=0.1, type=float,
                    help="the label smoothing epsilon for labeled data")

'''Configuration'''
parser.add_argument('--config_path', type=str, default=None, 
                    help='path to yaml config file, overwrites args')

    # return parser.parse_args()
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
# def generate_and_save_images(model, image, num_classes):
#     z = model.get_latent(image, training=False)
    
#     buf = io.BytesIO()
#     figure = plt.figure(figsize=(10, 2))
#     plt.subplot(1, num_classes+1, 1)
#     plt.imshow((image[0] + 1) / 2)
#     plt.title('original')
#     plt.axis('off')
#     for i in range(num_classes):
#         label = np.zeros((z.shape[0], num_classes))
#         label[:, i] = 1
#         xhat = model.decode_sample(z, label, training=False)
#         plt.subplot(1, num_classes+1, i+2)
#         plt.imshow((xhat[0] + 1) / 2)
#         plt.title('{}'.format(i))
#         plt.axis('off')
#     plt.savefig(buf, format='png')
#     # Closing the figure prevents it from being displayed directly inside the notebook.
#     plt.close(figure)
#     buf.seek(0)
#     # Convert PNG buffer to TF image
#     # Convert PNG buffer to TF image
#     image = tf.image.decode_png(buf.getvalue(), channels=4)
#     # Add the batch dimension
#     image = tf.expand_dims(image, 0)
#     return image

def generate_and_save_images(model, image, num_classes, step, save_dir):
    z = model.ae.z_encode(image, training=False)
    
    plt.figure(figsize=(10, 2))
    plt.subplot(1, num_classes+1, 1)
    plt.imshow(image[0])
    plt.title('original')
    plt.axis('off')
    for i in range(num_classes):
        label = np.zeros((z.shape[0], num_classes))
        label[:, i] = 1
        xhat = model.ae.decode(z, label, training=False)
        plt.subplot(1, num_classes+1, i+2)
        plt.imshow(xhat[0])
        plt.title('{}'.format(i))
        plt.axis('off')
    plt.savefig('{}/image_at_epoch_{}.png'.format(save_dir, step))
    # plt.show()
    plt.close()
#%%
def main():
    # '''argparse to dictionary'''
    # args = vars(get_args())
    '''argparse debugging'''
    args = vars(parser.parse_args(args=['--config_path', 'configs/cifar10_4000.yaml']))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

    '''dataset'''
    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    total_length = sum(1 for _ in datasetU)
    
    '''model'''
    model = VAE(args, num_classes)
    model.build(input_shape=(None, 32, 32, 3))
    # model.summary()
    
    buffer_model = VAE(args, num_classes)
    buffer_model.build(input_shape=(None, 32, 32, 3))
    # decay_model.set_weights(model.get_weights())
    
    '''optimizer'''
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args['lr_nf'], 
                                                                decay_steps=args['decay_steps'], 
                                                                decay_rate=args['decay_rate'])
    optimizer_nf = K.optimizers.Adam(args['lr_nf'], clipvalue=args['gradclip']) 

    '''
    <SGD + momentum + weight_decay>
    lambda: weight_decay parameter
    beta_1: momemtum
    lr: learning rate
    
    v(0) = 0
    for t in range(0, epochs):
        grad(t+1) = grad(t+1) + lambda * weight(t)
        v(t+1) = beta_1 * v(t) + grad(t+1)
        weight(t+1) = weight(t) - lr * v(t+1)
                    = weight(t) - lr * (beta_1 * v(t) + grad(t+1))
    '''
    
    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        if epoch == 0:
            '''warm-up'''
            lr = args['lr'] * 0.2
        # elif epoch < args['adjust_lr'][0]:
        #     lr = args['lr']
        # elif epoch < args['adjust_lr'][1]:
        #     lr = args['lr'] * 0.1
        # elif epoch < args['adjust_lr'][2]:
        #     lr = args['lr'] * 0.01
        # else:
        #     lr = args['lr'] * 0.001
        optimizer_nf.lr = lr_schedule(epoch)
        
        # if epoch % args['reconstruct_freq'] == 0:
        #     labeled_loss, unlabeled_loss, kl_y_loss, accuracy, sample_recon = train(datasetL, datasetU, model, buffer_model, lr, epoch, args, num_classes, total_length)
        # else:
        #     labeled_loss, unlabeled_loss, kl_y_loss, accuracy = train(datasetL, datasetU, model, buffer_model, lr, epoch, args, num_classes, total_length)
        loss, nf_loss, accuracy = train(datasetL, datasetU, model, buffer_model, lr, optimizer_nf, epoch, args, num_classes, total_length)
        val_nf_loss, val_recon_loss, val_elbo_loss, val_accuracy = validate(val_dataset, model, epoch, args, split='Validation')
        test_nf_loss, test_recon_loss, test_elbo_loss, test_accuracy = validate(test_dataset, model, epoch, args, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('nf_loss', nf_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
            # if epoch % args['reconstruct_freq'] == 0:
            #     tf.summary.image("train recon image", sample_recon, step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('nf_loss', val_nf_loss.result(), step=epoch)
            tf.summary.scalar('recon_loss', val_recon_loss.result(), step=epoch)
            tf.summary.scalar('elbo_loss', val_elbo_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar('nf_loss', test_nf_loss.result(), step=epoch)
            tf.summary.scalar('recon_loss', test_recon_loss.result(), step=epoch)
            tf.summary.scalar('elbo_loss', test_elbo_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        # Reset metrics every epoch
        loss.reset_states()
        nf_loss.reset_states()
        accuracy.reset_states()
        val_nf_loss.reset_states()
        val_recon_loss.reset_states()
        val_elbo_loss.reset_states()
        val_accuracy.reset_states()
        test_nf_loss.reset_states()
        test_recon_loss.reset_states()
        test_elbo_loss.reset_states()
        test_accuracy.reset_states()
        
        if epoch == 0:
            lr = args['lr']
            
        # if args['dataset'] == 'cifar10':
        #     if args['labeled_examples'] >= 2500:
        #         if epoch == args['adjust_lr'][0]:
        #             args['ewm'] = args['ewm'] * 5

    '''model & configurations save'''        
    model_path = f'{log_path}/{current_time}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save_weights(model_path + '/model_{}.h5'.format(current_time), save_format="h5")

    with open(model_path + '/args_{}.txt'.format(current_time), "w") as f:
        for key, value, in args.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
#%%
def train(datasetL, datasetU, model, buffer_model, lr, optimizer_nf, epoch, args, num_classes, total_length):
    loss_avg = tf.keras.metrics.Mean()
    nf_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    '''elbo part weight'''
    ew = weight_schedule(epoch, args['aew'], args['ewm'])
    '''un-supervised classification weight'''
    ucw = weight_schedule(epoch, round(args['wmf'] * args['epochs']), args['wrd'])

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)
    if args['dataset'] == 'cmnist':
        shuffle_and_batch2 = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=32, drop_remainder=False)

    iteratorL = iter(shuffle_and_batch(datasetL))
    if args['dataset'] == 'cmnist':
        iteratorU = iter(shuffle_and_batch2(datasetU))
    else:
        iteratorU = iter(shuffle_and_batch(datasetU))
        
    # iteration = (50000 - args['validation_examples']) // args['batch_size'] 
    iteration = total_length // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        '''augmentation'''
        try:
            imageL, labelL = next(iteratorL)
        except:
            if args['dataset'] == 'cmnist':
                iteratorL = iter(shuffle_and_batch2(datasetL))
            else:
                iteratorL = iter(shuffle_and_batch(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, _ = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(datasetU))
            imageU, _ = next(iteratorU)
            
        '''augmentation'''
        if args['augment']:
            imageL = augment(imageL)
            imageU = augment(imageU)
        
        '''mix-up weight'''
        mix_weight = [tf.constant(np.random.beta(args['epsilon'], args['epsilon'])), # labeled
                      tf.constant(np.random.beta(2.0, 2.0))] # unlabeled
        
        with tf.GradientTape(persistent=True) as tape:
            [[z, c, probU, xhatU], nf_args] = model(imageU, training=True)
            # z, c, probU, xhatU = model.ae(imageU, training=True) # unlabeled
            # z_ = tf.stop_gradient(z)
            # c_ = tf.stop_gradient(c)
            # nf_args = model.prior(z_, c_)
            probL = tf.nn.softmax(model.ae.c_encode(imageL, training=True), axis=-1)
            
            recon_lossU, cls_lossL, nf_loss = ELBO_criterion(args, imageU, xhatU, labelL, probL, nf_args)
            
            '''mix-up'''
            with tape.stop_recording():
                image_mixL, label_shuffleL = label_smoothing(imageL, labelL, mix_weight[0], tag='labeled')
                image_mixU, pseudo_labelU = label_smoothing(imageU, probU, mix_weight[1], tag='unlabeled')
            smoothed_probL = tf.nn.softmax(model.ae.c_encode(image_mixL, training=True), axis=-1)
            smoothed_probU = tf.nn.softmax(model.ae.c_encode(image_mixU, training=True), axis=-1)
            
            mixup_yL = - tf.reduce_mean(mix_weight[0] * tf.reduce_sum(label_shuffleL * tf.math.log(tf.clip_by_value(smoothed_probL, 1e-10, 1.0)), axis=-1))
            mixup_yL += - tf.reduce_mean((1. - mix_weight[0]) * tf.reduce_sum(labelL * tf.math.log(tf.clip_by_value(smoothed_probL, 1e-10, 1.0)), axis=-1))
            mixup_yU = - tf.reduce_mean(tf.reduce_sum(pseudo_labelU * tf.math.log(tf.clip_by_value(smoothed_probU, 1e-10, 1.0)), axis=-1))
            
            loss = (ew * recon_lossU) + mixup_yL + ucw * mixup_yU + cls_lossL

        '''AutoEncoder'''
        grads = tape.gradient(loss, model.ae.trainable_variables) 
        '''SGD + weight decay'''
        '''1. update gradient and buffer'''
        if (epoch == 0) and (batch_num == 0): # initialize buffer model weights only once
            for g, var, buffer_var in zip(grads, model.ae.trainable_variables, buffer_model.ae.trainable_variables):
                buffer_var.assign(g + tf.convert_to_tensor(args['wd'], tf.float32) * var)
        else:
            for g, var, buffer_var in zip(grads, model.ae.trainable_variables, buffer_model.ae.trainable_variables):
                buffer_var.assign(tf.convert_to_tensor(args['beta1'], tf.float32) * buffer_var + g + tf.convert_to_tensor(args['wd'], tf.float32) * var)
        '''2. update weight'''
        for var, buffer_var in zip(model.ae.trainable_variables, buffer_model.ae.trainable_variables):
                var.assign(var - tf.convert_to_tensor(lr, tf.float32) * buffer_var)
        
        '''Normalizing Flow'''
        grad = tape.gradient(nf_loss, model.prior.trainable_weights)
        optimizer_nf.apply_gradients(zip(grad, model.prior.trainable_weights))
        
        loss_avg(loss)
        nf_loss_avg(nf_loss)
        probL = tf.nn.softmax(model.ae.c_encode(imageL, training=False), axis=-1)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)

        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'NF': f'{nf_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
    
    # if epoch % args['reconstruct_freq'] == 0:
    #     sample_recon = generate_and_save_images(model, imageU[0][tf.newaxis, ...], num_classes)
    #     return labeled_loss_avg, unlabeled_loss_avg, kl_y_loss_avg, accuracy, sample_recon
    # else:
    #     return labeled_loss_avg, unlabeled_loss_avg, kl_y_loss_avg, accuracy
    return loss_avg, nf_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, num_classes, split):
    nf_loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()   
    elbo_loss_avg = tf.keras.metrics.Mean()   
    accuracy = tf.keras.metrics.Accuracy()

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        [[z, c, prob, xhat], nf_args] = model(image, training=True)
        recon_loss, cls_loss, nf_loss = ELBO_criterion(args, image, xhat, label, prob, nf_args)
        nf_loss_avg(nf_loss)
        recon_loss_avg(recon_loss)
        elbo_loss_avg(recon_loss + nf_loss + cls_loss)
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} ELBO Loss: {elbo_loss_avg.result():.4f}, NF: {nf_loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    
    return nf_loss_avg, recon_loss_avg, elbo_loss_avg, accuracy
#%%
def weight_schedule(epoch, epochs, weight_max):
    return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
if __name__ == '__main__':
    main()
#%%