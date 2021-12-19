#%%
import os

import tensorflow as tf
from tensorflow.python.ops.gen_list_ops import EmptyTensorList
import tensorflow_datasets as tfds
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
    
    return  train, test
#%%
def _list_to_tf_dataset(dataset):
    def _dataset_gen():
        for example in dataset:
            yield example
    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={'image':tf.uint8, 'label':tf.int64},
        output_shapes={'image': (32, 32, 3), 'label': ()}
    )
#%%
def split_dataset(dataset, num_labeled, num_validations, num_classes):
    dataset = dataset.shuffle(buffer_size=10000)
    counter = [0 for _ in range(num_classes)]
    labeled = []
    unlabeled = []
    validation = []
    for example in iter(dataset):
        label = int(example['label'])
        counter[label] += 1
        if counter[label] <= (num_labeled / num_classes):
            # labeled.append(example) 
            labeled.append({
                'image': example['image'],
                'label': example['label']
            })
            continue
        elif counter[label] <= (num_validations / num_classes + num_labeled / num_classes):
            # validation.append(example)
            validation.append({
                'image': example['image'],
                'label': example['label']
            })
        unlabeled.append({
            'image': example['image'],
            'label': tf.convert_to_tensor(-1, dtype=tf.int64)
        })
    labeled = _list_to_tf_dataset(labeled)
    unlabeled = _list_to_tf_dataset(unlabeled)
    validation = _list_to_tf_dataset(validation)
    return labeled, unlabeled, validation
#%%
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))
#%%
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#%%
def serialize_example(image, label):
    image = tf.image.encode_png(image)
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
#%%
def tf_serialize_example(example):
    tf_string = tf.py_function(
        serialize_example,
        (example['image'], example['label']),
        tf.string
    )
    return tf.reshape(tf_string, ())
#%%
def export_tfrecord_dataset(dataset_path, dataset):
    serialized_dataset = dataset.map(tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(dataset_path)
    writer.write(serialized_dataset)
#%%
def _parse_function(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)
#%%
def load_tfrecord_dataset(dataset_path):
    raw_dataset = tf.data.TFRecordDataset([dataset_path])
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset
#%%
def normalize_image(image):
    image = image / 255.
    return image
#%%
def process_parsed_dataset(dataset, num_classes):
    images = []
    labels = []
    for example in iter(dataset):
        decoded_image = tf.io.decode_png(example['image'], channels=3, dtype=tf.uint8)
        normalized_image = normalize_image(tf.cast(decoded_image, dtype=tf.float32))
        images.append(normalized_image)
        one_hot_label = tf.one_hot(example['label'], depth=num_classes, dtype=tf.float32)
        labels.append(one_hot_label)
    return tf.data.Dataset.from_tensor_slices({
        'image': images,
        'label': labels
    })
#%%
def fetch_dataset(args, log_dir):
    dataset_path = f'{log_dir}/datasets'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    num_classes = 100 if args['dataset'] == 'cifar100' else 10
    
    if any([not os.path.exists(f'{dataset_path}/{split}.tfrecord') for split in ['trainL', 'trainU', 'validation', 'test']]):
        train, test = download_dataset(dataset_name=args['dataset'])
        
        trainL, trainU, validation = split_dataset(dataset=train,
                                                   num_labeled=args['labeled_examples'],
                                                   num_validations=args['validation_examples'],
                                                   num_classes=num_classes)
        
        for name, dataset in [('trainL', trainL), ('trainU', trainU), ('validation', validation), ('test', test)]:
            # export_tfrecord_dataset(f'{dataset_path}/{name}.tfrecord', dataset)
            serialized_dataset = dataset.map(tf_serialize_example)
            writer = tf.io.TFRecordWriter(f'{dataset_path}/{name}.tfrecord')
            writer.write(serialized_dataset)
#%%