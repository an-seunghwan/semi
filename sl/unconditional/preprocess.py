#%%
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
from tqdm import tqdm
#%%
def download_dataset(dataset_name):
    train = None
    test = None
    if dataset_name == 'svhn':
        dataset = tfds.load(name='svhn_cropped')
        train = dataset['train']
        test = dataset['test']
    
    elif dataset_name == 'svhn+extra':
        dataset = tfds.load(name='svhn_cropped')
        train = dataset['train']
        train.concatenate(dataset['extra'])
        test = dataset['test']
    
    elif dataset_name == 'cifar10':
        dataset = tfds.load(name='cifar10')
        train = dataset['train']
        test = dataset['test']
    
    elif dataset_name == 'cifar100':
        dataset = tfds.load(name='cifar100')
        train = dataset['train']
        test = dataset['test']
        
    elif dataset_name == 'cmnist':
        dataset = tfds.load(name='mnist')
        train = dataset['train']
        test = dataset['test']
    
    return  train, test
#%%
def colored_mnist(image, args):
    '''
    FIXME
    '''
    tf.random.set_seed(args['seed'])
    image = tf.image.resize(image, [32, 32], method='nearest')
    
    if tf.random.uniform((1, 1)) > 0.5:
        # color
        image = tf.cast(image, tf.float32) / 255.
        color = np.random.uniform(0., 1., 3)
        color = color / np.linalg.norm(color)
        image = image * color[tf.newaxis, tf.newaxis, :]
        return image
    else:
        # edge detection
        image = cv2.Canny(image.numpy(), 10., 255.)
        image[np.where(image > 0)] = 1.
        image[np.where(image <= 0)] = 0.
        # color
        color = np.random.uniform(0., 1., 3)
        color = color / np.linalg.norm(color)
        image = image[..., tf.newaxis] * color[tf.newaxis, tf.newaxis, :]
        # width
        kernel = np.ones((1, 1))
        image = cv2.dilate(image, kernel)
        return image
#%%
def _list_to_tf_dataset(dataset, args):
    def _dataset_gen():
        for example in dataset:
            yield example
    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={'image':tf.float32, 'label':tf.int64} if args['dataset'] == 'cmnist' else {'image':tf.uint8, 'label':tf.int64},
        output_shapes={'image': (32, 32, 3), 'label': ()}
        # output_shapes={'image': (args['image_size'], args['image_size'], args['channel']), 'label': ()}
    )
#%%
def split_dataset(dataset, num_validations, num_classes, args):
    dataset = dataset.shuffle(buffer_size=10000, seed=args['seed'])
    counter = [0 for _ in range(num_classes)]
    labeled = []
    validation = []
    for example in tqdm(iter(dataset), desc='split_dataset'):
        label = int(example['label'])
        counter[label] += 1
        if counter[label] <= (num_validations / num_classes):
            validation.append({
                # 'image': example['image'],
                'image': colored_mnist(example['image'], args),
                'label': example['label']
            })
        labeled.append({
            # 'image': example['image'],
            'image': colored_mnist(example['image'], args),
            'label': tf.convert_to_tensor(-1, dtype=tf.int64)
        })
    labeled = _list_to_tf_dataset(labeled, args)
    validation = _list_to_tf_dataset(validation, args)
    return labeled, validation
#%%
def cmnist_test_dataset(dataset, args):
    test = []
    for example in tqdm(iter(dataset), desc='cmnist_test_dataset'):
        test.append({
            # 'image': example['image'],
            'image': colored_mnist(example['image'], args),
            'label': example['label']
        })
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
    if args['dataset'] == 'cmnist':
        image = image.astype(np.float32).tobytes()
    else:
        image = normalize_image(image.astype(np.float32)).tobytes()
    label = np.eye(num_classes).astype(np.float32)[label].tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    })) 
    return example.SerializeToString()
#%%
def cmnist_test_serialize_example(example, args):
    image = example['image']
    label = example['label']
    if args['dataset'] == 'cmnist':
        image = image.astype(np.float32).tobytes()
    else:
        image = normalize_image(image.astype(np.float32)).tobytes()
    label = label.astype(np.float32).tobytes()
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
def fetch_dataset(args, log_path):
    dataset_path = f'{log_path}/datasets'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    num_classes = 100 if args['dataset'] == 'cifar100' else 10
    
    if any([not os.path.exists(f'{dataset_path}/{split}.tfrecord') for split in ['train', 'validation', 'test']]):
        train, test = download_dataset(dataset_name=args['dataset'])
        
        train, validation = split_dataset(dataset=train,
                                        num_validations=args['validation_examples'],
                                        num_classes=num_classes,
                                        args=args)
        
        if args['dataset'] == 'cmnist':
            test = cmnist_test_dataset(dataset=test,
                                       args=args)
            
            for name, dataset in [('train', train), ('validation', validation), ('test', test)]:
                if name == 'test':
                    writer = tf.io.TFRecordWriter(f'{dataset_path}/{name}.tfrecord'.encode('utf-8'))
                    for x in tfds.as_numpy(dataset):
                        example = cmnist_test_serialize_example(x, args)
                        writer.write(example)
                else:
                    writer = tf.io.TFRecordWriter(f'{dataset_path}/{name}.tfrecord'.encode('utf-8'))
                    for x in tfds.as_numpy(dataset):
                        example = serialize_example(x, num_classes, args)
                        writer.write(example)
        else:
            for name, dataset in [('train', train), ('validation', validation), ('test', test)]:
                writer = tf.io.TFRecordWriter(f'{dataset_path}/{name}.tfrecord'.encode('utf-8'))
                for x in tfds.as_numpy(dataset):
                    example = serialize_example(x, num_classes, args)
                    writer.write(example)
    
    train = tf.data.TFRecordDataset(f'{dataset_path}/train.tfrecord'.encode('utf-8')).map(deserialize_example)
    validation = tf.data.TFRecordDataset(f'{dataset_path}/validation.tfrecord'.encode('utf-8')).map(deserialize_example)
    test = tf.data.TFRecordDataset(f'{dataset_path}/test.tfrecord'.encode('utf-8')).map(deserialize_example)
    
    return train, validation, test, num_classes
#%%