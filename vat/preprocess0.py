#%%
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%%
def download_dataset(dataset_name):
    
    assert dataset_name == 'cifar10'
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    """ZCA whitening"""
    datagen = ImageDataGenerator(zca_whitening=True)
    datagen.fit(x_train.astype('float32'))
    
    return datagen, x_train, y_train, x_test, y_test
#%%
def _list_to_tf_dataset(dataset, args):
    def _dataset_gen():
        for example in dataset:
            yield example
    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={'image':tf.uint8, 'label':tf.int64},
        output_shapes={'image': (32, 32, 3), 'label': ()}
        # output_shapes={'image': (args['image_size'], args['image_size'], args['channel']), 'label': ()}
    )
#%%
'''
- labeled and unlabeled are disjoint
- unlabeled partially includes validation
- labeled partially includes validation
'''
def split_dataset(dataset, num_labeled, num_validations, num_classes, args):
    np.random.seed(args['seed'])
    datagen, x_train, y_train = dataset
    counter = [0 for _ in range(num_classes)]
    labeled = []
    unlabeled = []
    validation = []
    count = 0
    for image, label in tqdm(datagen.flow(x_train, y_train, batch_size=1, seed=args['seed']), desc='split_dataset'):
        label = int(label)
        counter[label] += 1
        if counter[label] <= (num_validations / num_classes):
            validation.append({
                'image': image[0],
                'label': label
            })
        
        if counter[label] <= (num_labeled / num_classes):
            labeled.append({
                'image': image[0],
                'label': label
            })
        else:
            unlabeled.append({
                'image': image[0],
                'label': tf.convert_to_tensor(-1, dtype=tf.int64)
            })
        
        count += 1
        if count == x_train.shape[0]: break
        
    labeled = _list_to_tf_dataset(labeled, args)
    unlabeled = _list_to_tf_dataset(unlabeled, args)
    validation = _list_to_tf_dataset(validation, args)
    return labeled, unlabeled, validation
#%%
def split_dataset_test(dataset, args):
    np.random.seed(args['seed'])
    datagen, x_test, y_test = dataset
    test = []
    count = 0
    for image, label in tqdm(datagen.flow(x_test, y_test, batch_size=1, seed=args['seed']), desc='split_dataset_test'):
        label = int(label)
        test.append({
            'image': image[0],
            'label': label
        })
        count += 1
        if count == x_test.shape[0]: break
        
    test = _list_to_tf_dataset(test, args)
    return test
#%%
def normalize_image(image):
    image = image / 255.
    return image
#%%
def serialize_example(example, num_classes, args):
    image = example['image']
    label = example['label']
    image = normalize_image(image.astype(np.float32)).tobytes()
    label = np.eye(num_classes).astype(np.float32)[label].tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    })) 
    return example.SerializeToString()
#%%
def deserialize_example(serialized_string):
    image_feature_description = { 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.string), 
    } 
    example = tf.io.parse_single_example(serialized_string, image_feature_description) 
    image = tf.reshape(tf.io.decode_raw(example["image"], tf.float32), (32, 32, 3))
    label = tf.io.decode_raw(example["label"], tf.float32) 
    return image, label
#%%
# args = {
#     'seed': 1,
#     'dataset': 'cifar10',
#     'labeled_examples': 4000,
#     'validation_examples': 5000,
# }
# num_classes = 10
#%%
def fetch_dataset(args, log_path):
    dataset_path = f'{log_path}/datasets'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    num_classes = 100 if args['dataset'] == 'cifar100' else 10
    
    if any([not os.path.exists(f'{dataset_path}/{split}.tfrecord') for split in ['trainL', 'trainU', 'validation', 'test']]):
        datagen, x_train, y_train, x_test, y_test = download_dataset(dataset_name=args['dataset'])
        
        trainL, trainU, validation = split_dataset(dataset=[datagen, x_train, y_train],
                                                num_labeled=args['labeled_examples'],
                                                num_validations=args['validation_examples'],
                                                num_classes=num_classes,
                                                args=args)
        test = split_dataset_test(dataset=[datagen, x_test, y_test],
                                  args=args)
        
        for name, dataset in [('trainL', trainL), ('trainU', trainU), ('validation', validation), ('test', test)]:
            writer = tf.io.TFRecordWriter(f'{dataset_path}/{name}.tfrecord'.encode('utf-8'))
            for x in tfds.as_numpy(dataset):
                example = serialize_example(x, num_classes, args)
                writer.write(example)
    
    trainL = tf.data.TFRecordDataset(f'{dataset_path}/trainL.tfrecord'.encode('utf-8')).map(deserialize_example)
    trainU = tf.data.TFRecordDataset(f'{dataset_path}/trainU.tfrecord'.encode('utf-8')).map(deserialize_example)
    validation = tf.data.TFRecordDataset(f'{dataset_path}/validation.tfrecord'.encode('utf-8')).map(deserialize_example)
    test = tf.data.TFRecordDataset(f'{dataset_path}/test.tfrecord'.encode('utf-8')).map(deserialize_example)
    
    return trainL, trainU, validation, test, num_classes
#%%