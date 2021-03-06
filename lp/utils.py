#%%
from multiprocessing import cpu_count

import tensorflow as tf
import numpy as np
import faiss

from sklearn.preprocessing import normalize
import scipy
#%%
@tf.function
def augment(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, paddings=[(0, 0),
                            (4, 4),
                            (4, 4), 
                            (0, 0)], mode='REFLECT')
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    return x
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.trainable_variables, buffer_model.trainable_variables):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.trainable_variables, buffer_model.trainable_variables):
        buffer_var.assign(var)
#%%
def linear_rampup(current, lampup_length):
    if current >= lampup_length:
        return 1.
    else:
        return current / lampup_length
#%%
def cosine_rampdown(current, rampdown_length):
    return float(0.5 * (np.cos(np.pi * current / rampdown_length) + 1))
#%%
def build_pseudo_label(datasetL, datasetU, model, num_classes, args,
                       k=50, maxiter=20):
    print('extracting features...')
    autotune = tf.data.AUTOTUNE
    batch_iterL = lambda dataset: dataset.batch(batch_size=50, drop_remainder=False).prefetch(autotune)
    batch_iter = lambda dataset: dataset.batch(batch_size=args['batch_size'] - 50, drop_remainder=False).prefetch(autotune)
    iteratorL = iter(batch_iterL(datasetL))
    iteratorU = iter(batch_iter(datasetU))
    
    embeddings = []
    labelsL = []
    imagesU = [] # without augmentation and normalization
    
    '''normalization'''
    channel_stats = dict(mean=tf.reshape(tf.cast(np.array([0.4914, 0.4822, 0.4465]), tf.float32), (1, 1, 1, 3)),
                        std=tf.reshape(tf.cast(np.array([0.2470, 0.2435, 0.2616]), tf.float32), (1, 1, 1, 3)))
    
    while True:
        try:
            imageL, labelL = next(iteratorL)
            '''augmentation'''
            imageL = augment(imageL)
            '''normalization'''
            imageL -= channel_stats['mean']
            imageL /= channel_stats['std']
    
            _, feats = model(imageL, training=False)
            embeddings.append(feats)
            labelsL.append(labelL)
        except:
            break
    
    while True:
        try:
            imageU, _ = next(iteratorU)
            imagesU.append(imageU)
            '''augmentation'''
            imageU = augment(imageU)    
            '''normalization'''
            imageU -= channel_stats['mean']
            imageU /= channel_stats['std']

            _, feats = model(imageU, training=False)
            embeddings.append(feats)
        except:
            break
    
    embeddings = tf.concat(embeddings, axis=0)
    labelsL = tf.concat(labelsL, axis=0)
    imagesU = tf.concat(imagesU, axis=0)
    
    '''update pseudo-labels'''
    alpha = 0.99
    gamma = 3
    
    print('KNN searching...')
    embeddings = embeddings.numpy()
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(int(d))   
    # print(index.is_trained)
    # embeddings = normalize(embeddings, axis=1, norm='l2') # originally, normalize_L2 from faiss, inside of model
    index.add(embeddings)
    
    D, I = index.search(embeddings, k+1) 
    # D: cosine-similarity
    # I: topk index
    
    D = D[:, 1:] ** gamma
    I = I[:, 1:]
    
    print('building graph...')
    N = embeddings.shape[0]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T
    
    print('graph normalizing...')
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D
    
    print('diffusion...')
    Z = np.zeros((N, num_classes))
    I_alphaW = scipy.sparse.eye(N) - alpha * Wn
    labelsL = np.argmax(labelsL.numpy(), axis=1)
    for i in range(num_classes):
        cur_idx = np.where(labelsL == i)[0]
        y = np.zeros((N, ))
        '''implementation'''
        y[cur_idx] = 1. / len(cur_idx)
        f, _ = scipy.sparse.linalg.cg(I_alphaW, y, tol=1e-6, maxiter=maxiter)
        Z[:, i] = f
    Z[Z < 0] = 0 # handle numerical errors
    
    print('assigning pseudo labels and weights...')
    probs_Z = Z / np.sum(np.abs(Z), axis=1, keepdims=True)
    probs_Z[probs_Z < 0] = 0
    entropy = scipy.stats.entropy(probs_Z.T)
    # entropy = -np.sum(probs_Z * np.log(np.clip(probs_Z, 1e-10, 1.)), axis=1)
    weights = 1. - entropy / np.log(num_classes)
    weights = weights / np.max(weights)
    plabels = np.argmax(probs_Z, axis=1)
    
    '''labeled accuracy using label propagation'''
    accL = len(np.where((plabels[:tf.shape(labelsL)[0]] - labelsL) == 0)[0]) / len(labelsL)
    
    '''replace labeled examples with true values'''
    plabels[:tf.shape(labelsL)[0]] = labelsL
    weights[:tf.shape(labelsL)[0]] = 1.
    
    print('assigning class weights...')
    class_weights = np.zeros((1, num_classes))
    for i in range(num_classes):
        cur_idx = np.where(plabels == i)[0]
        '''implementation'''
        class_weights[0, i] = (N / num_classes) / len(cur_idx)
    
    print('build pseudo-label dataset...')
    plabels = tf.one_hot(plabels[tf.shape(labelsL)[0]:], depth=num_classes)
    pseudo_datasetU = tf.data.Dataset.from_tensor_slices((
        # images.numpy()[tf.shape(labelsL)[0]:], 
        imagesU, 
        plabels, 
        tf.cast(weights[tf.shape(labelsL)[0]:], tf.float32)))
    return pseudo_datasetU, class_weights, accL
#%%