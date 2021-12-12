import os
import random
import tempfile
from random import seed

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold


def sub_sample(data, n):
    data_ = data.copy()
    data_.index = np.arange(data_.shape[0])
    return data.loc[:n-1, data_.columns]


def train_val_split(input_data, output_data=None, meta_data=None, n_parallel_models=1, do_kfolds=False, val_ratio=0.2,
                    shuffle=True, seed_val=None, return_tfrecord=False, tfrecord_name=None, batch_size=None):
    """ Train validation split.

        Parameters:
            input_data (list[array], array): Input data.
            output_data (list[array], array): Output data.
            meta_data (list[array], array): Meta data.
            n_parallel_models (int): Number of parallel models.
            do_kfolds (bool): Whether to do kfolds for cross-validation or not.
            val_ratio (float): Ratio for validation.
            shuffle (bool): Whether to shuffle or not.
            seed_val (int): Seed value.
            return_tfrecord (bool): Whether to return tfrecord or not.
            tfrecord_path (str): Name of the tfrecord.
            batch_size (int): Batch size used for the tfrecord dataset.

        Returns:
            4 list[np.ndarray] or list[tfrecord].
    """
    n_samples = input_data.shape[0]
    distributions = ['train', 'val']
    data = dict(x=input_data if isinstance(input_data, list) else [input_data])
    split_data = dict(x={distribution: [] for distribution in distributions})
    if output_data is not None:
        data.update(y=output_data if isinstance(output_data, list) else [output_data])
        split_data.update(y={distribution: [] for distribution in distributions})
    if meta_data is not None:
        data.update(meta=meta_data if isinstance(meta_data, list) else [meta_data])
        split_data.update(meta={distribution: [] for distribution in distributions})
    if do_kfolds and n_parallel_models > 1:
        kf = KFold(
            n_splits=n_parallel_models,
            shuffle=True,
            random_state=seed_val
        )
        n_train = min([data_[0].shape[0] for data_ in kf.split(range(n_samples))])
        n_val = min([data_[1].shape[0] for data_ in kf.split(range(n_samples))])
        inds = [dict(train=train_inds[:n_train, ...], val=val_inds[:n_val, ...])
                for train_inds, val_inds in kf.split(range(n_samples))]
    else:
        inds = list(np.arange(n_samples))
        if shuffle:
            random.shuffle(inds, random=seed(seed_val))
        line = int(len(inds) * (1 - val_ratio))
        inds = [dict(train=inds[:line], val=inds[line:]) for _ in np.arange(0, n_parallel_models)]
    for inds_ in inds:
        for distribution in distributions:
            x_ = []
            for sub_x in data['x']:
                x_ += [sub_x[inds_[distribution], ...]]
            split_data['x'][distribution] += [x_]
            if output_data is not None:
                y_ = []
                for sub_y in data['y']:
                    y_ += [sub_y[inds_[distribution], ...]]
                split_data['y'][distribution] += [y_]
            if meta_data is not None:
                meta_ = []
                for sub_meta in data['meta']:
                    meta_ += [sub_meta[inds_[distribution], ...]]
                split_data['meta'][distribution] += [meta_]
    if return_tfrecord:
        if tfrecord_name is None:
            tfrecord_name = os.path.join(tempfile.gettempdir(), 'tfrecord')
        for split_data_name in split_data.keys():
            for distribution in distributions:
                for i in range(n_parallel_models):
                    for j in range(split_data[split_data_name][distribution][i]):
                        tfrecord_name_ = '_'.join([tfrecord_name, split_data_name, distribution, str(i)]) + '.tfrecords'
                        write_tfrecord(
                            data=split_data[split_data_name][distribution][i][j],
                            name=tfrecord_name_
                        )
                        split_data[split_data_name][distribution][i] = [read_tfrecord(tfrecord_name_)]
    returns = []
    for split_data_name in split_data.keys():
        for distribution in distributions:
            return_ = split_data[split_data_name][distribution]
            if n_parallel_models == 1:
                return_ = return_[0]
            returns += [return_]
    returns += [inds]
    return returns


def write_tfrecord(data, name):
    with tf.io.TFRecordWriter(name) as writer:
        for i in range(len(data)):
            tf_example = example(data[i])
            writer.write(tf_example.SerializeToString())


def read_tfrecord(name, batch_size=None):
    dataset = tf.data.TFRecordDataset(name)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_function)
    return dataset


def parse_function(example_proto):
    feature_description = {'data': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example_proto, feature_description)
    data = example['data']
    tf.image.decode_jpeg()
    return


def example(data):
    feature = {'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()]))}
    return tf.train.Example(features=tf.train.Features(feature=feature))


def to_time_series(dataset, targets, look_back=1):
    union_dataset = np.concatenate((dataset, targets), axis=-1)
    x, y = [], []
    for i in range(len(union_dataset)-look_back-1):
        a = union_dataset[i:(i+look_back), ...]
        x.append(a)
        y.append(union_dataset[i + look_back, -1])
    return np.array(x), np.expand_dims(np.array(y), axis=-1)
