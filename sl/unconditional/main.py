#%%
'''

'''
#%%
import argparse
import os

os.chdir(r'D:\semi\sl\unconditional') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/unconditional') # main directory (repository)

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
# from criterion import ELBO_criterion
from mixup import augment
#%%
config = tf.compat.v1.ConfigProto()
'''
GPU 메모리를 전부 할당하지 않고, 아주 적은 비율만 할당되어 시작됨
프로세스의 메모리 수요에 따라 자동적으로 증가
but
GPU 메모리를 처음부터 전체 비율을 사용하지 않음
'''
config.gpu_options.allow_growth = True

# '''
# 분산 학습 설정
# '''
# strategy = tf.distribute.MirroredStrategy()
# session = tf.compat.v1.InteractiveSession(config=config)
#%%
# import ast
# def arg_as_list(s):
#     v = ast.literal_eval(s)
#     if type(v) is not list:
#         raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
#     return v
#%%
def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results (ex. generating color MNIST)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra, cmnist)')
    # parser.add_argument("--image_size", default=32, type=int,
    #                     metavar='Image Size', help='the size of height or width for image')
    # parser.add_argument("--channel", default=3, type=int,
    #                     metavar='Channel size', help='the size of image channel')
    parser.add_argument('--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=600, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=50, type=int,
                        metavar='N', help='reconstruct frequency (default: 50)')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters (Encoder and Decoder)'''
    # parser.add_argument('--net-name', default="wideresnet-28-2", type=str, help="the name for network to use")
    parser.add_argument('--depth', type=int, default=28, 
                        help='depth for WideResnet (default: 28)')
    parser.add_argument('--width', type=int, default=2, 
                        help='widen factor for WideResnet (default: 2)')
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='slope parameter for LeakyReLU (default: 0.1)')
    parser.add_argument('-dr', '--drop_rate', default=0, type=float, 
                        help='drop rate for the network')
    parser.add_argument("--br", "--bce_reconstruction", action='store_true', 
                        help='Do BCE Reconstruction')
    parser.add_argument('--x_sigma', default=1, type=float,
                        help="The standard variance for reconstructed images, work as regularization")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=128, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    # parser.add_argument('--cmi', "--continuous_mutual_info", default=0, type=float,
    #                     help='The mutual information bounding between x and the continuous variable z')
    # parser.add_argument('--dmi', "--discrete_mutual_info", default=0, type=float,
    #                     help='The mutual information bounding between x and the discrete variable z')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--lambda1', default=5., type=float,
                        help="adjust classification loss weight")
    parser.add_argument('--lambda2', default=10., type=float,
                        help="adjust mutual information loss weight")
    # parser.add_argument('--ewm', '--elbo-weight-max', default=1e-3, type=float, 
    #                     metavar='weight for elbo loss part')
    # parser.add_argument('--aew', '--adjust-elbo-weight', default=400, type=int,
    #                     metavar="the epoch to adjust elbo weight to max")
    # parser.add_argument('--wrd', default=1, type=float,
    #                     help="the max weight for the optimal transport estimation of discrete variable c")
    # parser.add_argument('--wmf', '--weight-modify-factor', default=0.4, type=float,
    #                     help="weight  will get wrz at amf * epochs")
    # parser.add_argument('--pwm', '--posterior-weight-max', default=1, type=float,
    #                     help="the max value for posterior weight")
    # parser.add_argument('--apw', '--adjust-posterior-weight', default=200, type=float,
    #                     help="adjust posterior weight")

    '''Optimizer Parameters (Encoder and Decoder)'''
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    # parser.add_argument('--beta1', default=0.9, type=float, metavar='Beta1 In ADAM and SGD',
    #                     help='beta1 for adam as well as momentum for SGD')
    # parser.add_argument('--adjust_lr', default=[400, 500, 550], type=arg_as_list,
    #                     help="The milestone list for adjust learning rate")
    # parser.add_argument('--weight_decay', default=5e-4, type=float)

    # '''Optimizer Transport Estimation Parameters'''
    # parser.add_argument('--epsilon', default=0.1, type=float,
    #                     help="the label smoothing epsilon for labeled data")
    # # parser.add_argument('--om', action='store_true', help="the optimal match for unlabeled data mixup")
    
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
    parser.add_argument('--lr_nf', '--learning-rate-nf', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate for normalizing flow')
    parser.add_argument('--reg', default=0.01, type=float,
                        help='L2 regularization parameter for dense layers in Real NVP')
    parser.add_argument('--decay_steps', default=1, type=int,
                        help='decay steps for exponential decay schedule')
    parser.add_argument('--decay_rate', default=0.95, type=float,
                        help='decay rate for exponential decay schedule')
    parser.add_argument('--gradclip', default=1., type=float,
                        help='gradclip value')

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
def generate_and_save_images(model, image, num_classes):
    z = model.ae.z_encode(image, training=False)
    
    buf = io.BytesIO()
    figure = plt.figure(figsize=(10, 2))
    plt.subplot(1, num_classes+1, 1)
    plt.imshow((image[0] + 1) / 2)
    plt.title('original')
    plt.axis('off')
    for i in range(num_classes):
        label = np.zeros((z.shape[0], num_classes))
        label[:, i] = 1
        xhat = model.ae.decode(z, label, training=False)
        plt.subplot(1, num_classes+1, i+2)
        plt.imshow((xhat[0] + 1) / 2)
        plt.title('{}'.format(i))
        plt.axis('off')
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
#%%
def main():
    '''argparse to dictionary'''
    args = vars(get_args())
    # '''argparse debugging'''
    # args = vars(parser.parse_args(args=['--config_path', 'configs/cmnist.yaml']))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}'

    dataset, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    
    model = VAE(args, num_classes)
    model.build(input_shape=(None, 32, 32, 3))
    # model.summary()
    
    # decay_model = VAE(args, num_classes)
    # decay_model.build(input_shape=(None, 32, 32, 3))
    # decay_model.set_weights(model.get_weights())

    '''
    <SGD + momentum + weight_decay>
    \lambda: weight_decay parameter
    \beta_1: momemtum
    
    v(0) = 0
    for i in range(epochs):
        grad(i+1) = grad(i) + \lambda * weight
        v(i+1) = \beta_1 * v(i) + grad(i+1)
        weight(i+1) = weight(i) - lr * v(i+1)
    
    weight(i+1) 
    = weight(i) - lr * (\beta_1 * v(i) + grad(i+1))
    = weight(i) - lr * (\beta_1 * v(i) + grad(i) + \lambda * weight)
    = weight(i) - lr * (\beta_1 * v(i) + grad(i)) - lr * \lambda * weight
    
    SGD + momentum : weight(i) - lr * (\beta_1 * v(i) + grad(i))
    weight_decay : - lr * \lambda * weight
    '''
    optimizer = K.optimizers.Adam(learning_rate=args['lr'])
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args['lr_nf'], 
                                                                decay_steps=args['decay_steps'], 
                                                                decay_rate=args['decay_rate'])
    optimizer_nf = K.optimizers.Adam(args['lr_nf'], clipvalue=args['gradclip']) 

    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        optimizer_nf.lr = lr_schedule(epoch)
        # if epoch == 0:
        #     '''warm-up'''
        #     optimizer.lr = args['lr'] * 0.2
        # elif epoch < args['adjust_lr'][0]:
        #     optimizer.lr = args['lr']
        # elif epoch < args['adjust_lr'][1]:
        #     optimizer.lr = args['lr'] * 0.1
        # elif epoch < args['adjust_lr'][2]:
        #     optimizer.lr = args['lr'] * 0.01
        # else:
        #     optimizer.lr = args['lr'] * 0.001
        
        if epoch % args['reconstruct_freq'] == 0:
            loss, nf_loss, accuracy, sample_recon = train(dataset, model, optimizer, optimizer_nf, epoch, args, num_classes)
        else:
            loss, nf_loss, accuracy = train(dataset, model, optimizer, optimizer_nf, epoch, args, num_classes)
        val_nf_loss, val_recon_loss, val_elbo_loss, val_accuracy = validate(val_dataset, model, epoch, args, split='Validation')
        test_nf_loss, test_recon_loss, test_elbo_loss, test_accuracy = validate(test_dataset, model, epoch, args, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('nf_loss', nf_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
            if epoch % args['reconstruct_freq'] == 0:
                tf.summary.image("train recon image", sample_recon, step=epoch)
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

    '''model & configurations save'''        
    model_path = f'{log_path}/{current_time}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save_weights(model_path + '/model_{}.h5'.format(current_time), save_format="h5")

    with open(model_path + '/args_{}.txt'.format(current_time), "w") as f:
        for key, value, in args.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
            
    # if epoch == 0:
    #     optimizer.lr = args['lr']
        
    # if args['dataset'] == 'cifar10':
    #     if args['labeled_examples'] >= 2500:
    #         if epoch == args['adjust_lr'][0]:
    #             args['ewm'] = args['ewm'] * 5
#%%
def train(dataset, model, optimizer, optimizer_nf, epoch, args, num_classes):
    loss_avg = tf.keras.metrics.Mean()
    nf_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # '''elbo part weight'''
    # ew = weight_schedule(epoch, args['aew'], args['ewm'])
    # '''mix-up parameters'''
    # pwm = weight_schedule(epoch, args['apw'], args['pwm'])
    # '''un-supervised classification weight'''
    # ucw = weight_schedule(epoch, round(args['wmf'] * args['epochs']), args['wrd'])

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)

    iterator = iter(shuffle_and_batch(dataset))
    
    iteration = 50000 // args['batch_size'] 
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        '''augmentation'''
        image, label = next(iterator)
        image = augment(image)
        
        with tf.GradientTape(persistent=True) as tape:
            [[_, _, prob, xhat], nf_args] = model(image)
            '''reconstruction'''
            if args['br']:
                recon_loss = tf.reduce_mean(- tf.reduce_sum(image * tf.math.log(xhat) + 
                                                            (1. - image) * tf.math.log(1. - xhat), axis=[1, 2, 3]))
            else:
                recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(xhat - image) / (2. * (args['x_sigma'] ** 2)), axis=[1, 2, 3]))
                
            '''classification'''
            cls_loss = tf.reduce_mean(- tf.reduce_sum(label * tf.math.log(prob), axis=-1))
            
            '''mutual information'''
            c_recon = model.ae.c_encode(xhat, training=True)
            prob_recon = tf.nn.softmax(c_recon, axis=-1)
            info = tf.reduce_mean(- tf.reduce_sum(label * tf.math.log(prob_recon), axis=-1))
            
            loss = recon_loss + args['lambda1'] * cls_loss + args['lambda2'] * info
            
            '''prior'''
            z_nf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(nf_args[0] - 0) / 2., axis=1))
            z_nf_loss -= tf.reduce_mean(nf_args[1], axis=-1)
            c_nf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(nf_args[2] - 0) / 2., axis=1))
            c_nf_loss -= tf.reduce_mean(nf_args[3], axis=-1)
            nf_loss = z_nf_loss + c_nf_loss

        grads = tape.gradient(loss, model.ae.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.ae.trainable_variables)) 
        # weight_decay(model.ae, decay_model.ae, decay_rate=args['weight_decay'] * args['lr']) # weight decay
        
        grad = tape.gradient(nf_loss, model.prior.trainable_weights)
        optimizer_nf.apply_gradients(zip(grad, model.prior.trainable_weights))
        
        loss_avg(loss)
        nf_loss_avg(nf_loss)
        _, _, prob, _ = model.ae(image, training=False)
        accuracy(tf.argmax(label, axis=1, output_type=tf.int32), prob)

        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'NF Loss': f'{nf_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
    
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images(model, image[0][tf.newaxis, ...], num_classes)
        return loss_avg, nf_loss_avg, accuracy, sample_recon
    else:
        return loss_avg, nf_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, split):
    nf_loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()   
    elbo_loss_avg = tf.keras.metrics.Mean()   
    accuracy = tf.keras.metrics.Accuracy()

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        [[_, _, prob, xhat], nf_args] = model(image, training=False)
        '''reconstruction'''
        if args['br']:
            recon_loss = tf.reduce_mean(- tf.reduce_sum(image * tf.math.log(xhat) + 
                                                        (1. - image) * tf.math.log(1. - xhat), axis=[1, 2, 3]))
        else:
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(xhat - image) / (2. * (args['x_sigma'] ** 2)), axis=[1, 2, 3]))
            
        '''classification'''
        cls_loss = tf.reduce_mean(- tf.reduce_sum(label * tf.math.log(prob), axis=-1))
        
        '''prior'''
        z_nf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(nf_args[0] - 0) / 2., axis=1))
        z_nf_loss -= tf.reduce_mean(nf_args[1], axis=-1)
        c_nf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(nf_args[2] - 0) / 2., axis=1))
        c_nf_loss -= tf.reduce_mean(nf_args[3], axis=-1)
        nf_loss = z_nf_loss + c_nf_loss
        
        nf_loss_avg(nf_loss)
        recon_loss_avg(recon_loss)
        elbo_loss_avg(recon_loss + nf_loss + cls_loss)
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} ELBO Loss: {elbo_loss_avg.result():.4f}, {split} Recon: {recon_loss_avg.result():.4f}, {split} Prior: {nf_loss_avg.result():.4f}, {split} Accuracy: {accuracy.result():.3%}')
    
    return nf_loss_avg, recon_loss_avg, elbo_loss_avg, accuracy
#%%
# def weight_schedule(epoch, epochs, weight_max):
#     return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
if __name__ == '__main__':
    main()
#%%