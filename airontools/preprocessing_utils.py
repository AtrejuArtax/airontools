import os
import random
import tempfile
from random import seed
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from numpy.typing import NDArray
from sklearn.model_selection import KFold
from tensorflow import DType


def to_time_series(
    dataset: NDArray, targets: NDArray, look_back: int = 1
) -> Tuple[NDArray, NDArray]:
    """To time series. It assumes the data is sequentially ordered, i.e. the first raw is the most recent sample in
    time.

    Parameters:
        dataset (NDArray): Dataset.
        targets (NDArray): Targets.
        look_back (int): Meta data.

    Returns:
        2 NDArray, one for the data and the other one for the targets.
    """
    union_dataset = np.concatenate((dataset, targets), axis=-1)
    x, y = [], []
    for i in range(len(union_dataset) - look_back - 1):
        x.append(union_dataset[i : (i + look_back), ...])
        y.append(targets[i + look_back, ...])
    return np.array(x), np.array(y)


def write_tfrecord(dataset: NDArray, filepath: str) -> None:
    """Write tensorflow record.

    Parameters:
        dataset (NDArray): Dataset.
        filepath (str): File path.
    """
    with tf.io.TFRecordWriter(filepath) as writer:
        for i in range(len(dataset)):
            sample = dataset[i].reshape((np.prod(dataset[i].shape),)).astype(np.float32)
            example = _example(sample)
            writer.write(example.SerializeToString())


def read_tfrecord(
    filepath: str, sample_shape: tuple, dtype: tf.DType = tf.float32
) -> NDArray:
    """Read tensorflow record.

    Parameters:
        filepath (str): File path.
        sample_shape: Sample shape.
        dtype (tf.DType): Data type.

    Returns:
        An NDArray.
    """
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(parse_function(sample_shape=sample_shape, dtype=dtype))
    return dataset


def parse_function(sample_shape: Tuple[int], dtype: DType):
    """Parse function.

    Parameters:
        sample_shape: Sample shape.
        dtype (tf.DType): Data type.

    Returns:
        A function.
    """

    def parse_function_(record):
        feature_description = {"data": tf.io.FixedLenFeature([], tf.string)}
        data = tf.io.parse_single_example(record, feature_description)
        data = tf.io.decode_raw(data["data"], out_type=dtype)
        data = tf.reshape(data, sample_shape)
        return data

    return parse_function_


def train_val_split(
    input_data: Union[List[Union[NDArray, tf.data.Dataset]], NDArray, tf.data.Dataset],
    output_data: Optional[
        List[Union[NDArray, tf.data.Dataset]], NDArray, tf.data.Dataset
    ] = None,
    meta_data: Optional[
        List[Union[NDArray, tf.data.Dataset]], NDArray, tf.data.Dataset
    ] = None,
    n_parallel_models: int = 1,
    do_kfolds: bool = False,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed_val: int = None,
    return_tfrecord: bool = False,
    tfrecord_name: str = None,
):
    """Train validation split.

    Parameters:
        input_data (Union[List[Union[NDArray, tf.data.Dataset]], NDArray, tf.data.Dataset]): Input data.
        output_data (Optional[List[Union[NDArray, tf.data.Dataset]], NDArray, tf.data.Dataset]): Output data.
        meta_data (Optional[List[Union[NDArray, tf.data.Dataset]], NDArray, tf.data.Dataset]): Meta data.
        n_parallel_models (int): Number of parallel models.
        do_kfolds (bool): Whether to do kfolds for cross-validation or not.
        val_ratio (float): Ratio for validation.
        shuffle (bool): Whether to shuffle or not.
        seed_val (int): Seed value.
        return_tfrecord (bool): Whether to return tfrecord or not.
        tfrecord_name (str): Name of the tfrecord.

    Returns:
        4 list[Union[NDArray, tf.data.Dataset]].
    """
    # ToDo: break the function into smaller functions.
    distributions = ["train", "val"]
    data = dict(x=_to_list_array(input_data))
    n_samples = data["x"][0].shape[0]
    split_data = dict(x={distribution: [] for distribution in distributions})
    if output_data is not None:
        data.update(y=_to_list_array(output_data))
        split_data.update(y={distribution: [] for distribution in distributions})
    if meta_data is not None:
        data.update(meta=_to_list_array(meta_data))
        split_data.update(meta={distribution: [] for distribution in distributions})
    if do_kfolds and n_parallel_models > 1:
        kf = KFold(n_splits=n_parallel_models, shuffle=True, random_state=seed_val)
        n_train = min(data_[0].shape[0] for data_ in kf.split(range(n_samples)))
        n_val = min(data_[1].shape[0] for data_ in kf.split(range(n_samples)))
        inds = [
            dict(train=train_inds[:n_train, ...], val=val_inds[:n_val, ...])
            for train_inds, val_inds in kf.split(range(n_samples))
        ]
    else:
        inds = list(np.arange(n_samples))
        if shuffle:
            random.shuffle(inds, random=seed(seed_val))
        line = int(len(inds) * (1 - val_ratio))
        inds = [dict(train=inds[:line], val=inds[line:])] * n_parallel_models
    for inds_ in inds:
        for distribution in distributions:
            x_ = []
            for sub_x in data["x"]:
                x_ += [sub_x[inds_[distribution], ...]]
            split_data["x"][distribution] += [x_]
            if output_data is not None:
                y_ = []
                for sub_y in data["y"]:
                    y_ += [sub_y[inds_[distribution], ...]]
                split_data["y"][distribution] += [y_]
            if meta_data is not None:
                meta_ = []
                for sub_meta in data["meta"]:
                    meta_ += [sub_meta[inds_[distribution], ...]]
                split_data["meta"][distribution] += [meta_]
    if return_tfrecord:
        if tfrecord_name is None:
            tfrecord_name = os.path.join(tempfile.gettempdir(), "tfrecord")
        for name in split_data.keys():
            for distribution in distributions:
                for i in range(n_parallel_models):
                    for j in range(len(split_data[name][distribution][i])):
                        tfrecord_name_ = "_".join(
                            [tfrecord_name, name, distribution, "fold", str(i), str(j)],
                        )
                        _data = split_data[name][distribution][i][j]
                        sample_shape = tuple(_data.shape[1:])
                        write_tfrecord(
                            dataset=_data, filepath=tfrecord_name_ + ".tfrecords"
                        )
                        split_data[name][distribution][i][j] = read_tfrecord(
                            filepath=tfrecord_name_ + ".tfrecords",
                            sample_shape=sample_shape,
                        )
    returns = []
    for name in split_data.keys():
        for distribution in distributions:
            return_ = split_data[name][distribution]
            if len(data[name]) == 1:
                return_ = [_return[0] for _return in return_]
            if n_parallel_models == 1:
                return_ = return_[0]
            returns += [return_]
    returns += [inds]
    return returns


def _example(data: np.array):
    feature = {"data": _bytes_feature(data.tobytes())}
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    return example


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _to_list_array(data):
    data_list = []
    if not isinstance(data, list):
        data = [data]
    for i in range(len(data)):
        if isinstance(data[i], tf.data.Dataset):
            data[i] = tfds.as_numpy(data[i])
            for _, data_ in data[i].items():
                data_list += [data_]
        else:
            data_list += [data[i]]
    return data_list
