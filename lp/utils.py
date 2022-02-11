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
    for var, buffer_var in zip(model.variables, buffer_model.variables):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.variables, buffer_model.variables):
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
    '''extract features'''
    batch_iter = lambda dataset: dataset.batch(batch_size=args['batch_size'], drop_remainder=False)
    iteratorL = iter(batch_iter(datasetL))
    iteratorU = iter(batch_iter(datasetU))
    
    embeddings = []
    labelsL = []
    images = []
    
    while True:
        try:
            imageL, labelL = next(iteratorL)
            _, feats = model(imageL, training=False)
            embeddings.append(feats)
            labelsL.append(labelL)
            images.append(imageL)
        except:
            break
    
    while True:
        try:
            imageU, _ = next(iteratorU)
            _, feats = model(imageU, training=False)
            embeddings.append(feats)
            images.append(imageU)
        except:
            break
    
    embeddings = tf.concat(embeddings, axis=0)
    labelsL = tf.concat(labelsL, axis=0)
    images = tf.concat(images, axis=0)
    
    # shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=len(embeddings), dtype=tf.int32))
    # embeddings = tf.gather(embeddings, shuffled_indices)
    # labels = tf.gather(labels, shuffled_indices)
    # tf.where(tf.less(shuffled_indices, tf.shape(labelsL)[0]), tf.cast(shuffled_indices, tf.float32), -1.0)
    
    '''update pseudo-labels'''
    alpha = 0.99
    gamma = 3
    
    '''KNN search'''
    embeddings = embeddings.numpy()
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(int(d))   
    # print(index.is_trained)
    # embeddings = normalize(embeddings, axis=1, norm='l2') # originally, normalize_L2 from faiss
    index.add(embeddings)
    
    D, I = index.search(embeddings, k+1) 
    # D: L2 distance
    # I: topk index
    
    D = D[:, 1:] ** gamma
    I = I[:, 1:]
    
    '''create graph'''
    N = embeddings.shape[0]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T
    
    '''normalize graph'''
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D
    
    '''initialize Y (normalize with the class size)'''
    Z = np.zeros((N, num_classes))
    I_alphaW = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    labelsL = np.argmax(labelsL.numpy(), axis=1)
    for i in range(num_classes):
        cur_idx = np.where(labelsL == i)[0]
        y = np.zeros((N, ))
        y[cur_idx] = 1. / len(cur_idx)
        f, _ = scipy.sparse.linalg.cg(I_alphaW, y, tol=1e-6, maxiter=maxiter)
        Z[:, i] = f
    Z[Z < 0] = 0 # handle numerical errors
    
    '''compute weight based on entropy'''
    probs_Z = Z / np.sum(Z, axis=1, keepdims=True)
    probs_Z[probs_Z < 0] = 0
    plabels = np.argmax(probs_Z, axis=1)
    entropy = -np.sum(probs_Z * np.log(np.clip(probs_Z, 1e-10, 1.)), axis=1)
    weights = 1. - entropy / np.log(num_classes)
    weights = weights / np.max(weights)
    
    '''replace labeled examples with true values'''
    plabels[:tf.shape(labelsL)[0]] = labelsL
    weights[:tf.shape(labelsL)[0]] = 1.
    
    '''compute weight for each class'''
    class_weights = np.zeros((1, num_classes))
    for i in range(num_classes):
        cur_idx = np.where(plabels == i)[0]
        class_weights[0, i] = (N / num_classes) / len(cur_idx)
    
    '''build pseudo-label dataset'''
    plabels = tf.one_hot(plabels[tf.shape(labelsL)[0]:], depth=num_classes)
    pseudo_datasetU = tf.data.Dataset.from_tensor_slices((images.numpy()[tf.shape(labelsL)[0]:], 
                                                          plabels, 
                                                          tf.cast(weights[tf.shape(labelsL)[0]:, None], tf.float32)))
    return pseudo_datasetU, class_weights
#%%