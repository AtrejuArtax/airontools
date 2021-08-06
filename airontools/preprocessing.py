import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import KFold
import random
from random import seed


def sub_sample(data, n):
    data_ = data.copy()
    data_.index = np.arange(data_.shape[0])
    return data.loc[:n-1, data_.columns]


def array_to_list(input_data, output_data, n_parallel_models, do_kfolds=False, val_ratio=0.2, shuffle=True,
                  seed_val=None):
    """ From array to list of numpy.

        Parameters:
            input_data (np.ndarray): Input data.
            output_data (np.ndarray): Output data.
            n_parallel_models (int): Number of parallel models.
            do_kfolds (bool): Whether to do kfolds for cross-validation or not.
            val_ratio (float): Ratio for validation.
            shuffle (bool): Whether to shuffle or not.
            seed_val (int): Seed value.

        Returns:
            4 lists.
    """
    x_train, x_val, y_train, y_val = [], [], [], []
    if do_kfolds and n_parallel_models > 1:
        kf = KFold(n_splits=n_parallel_models, shuffle=True, random_state=seed_val)
        n_train = min([data[0].shape[0] for data in kf.split(range(input_data.shape[0]))])
        n_val = min([data[1].shape[0] for data in kf.split(range(input_data.shape[0]))])
        train_val_inds = [[train_inds[:n_train, ...], val_inds[:n_val, ...]]
                          for train_inds, val_inds in kf.split(range(input_data.shape[0]))]
    else:
        inds = np.arange(0, input_data.shape[0])
        if shuffle:
            random.shuffle(inds, random=seed(seed_val))
        line = int(len(inds) * (1 - val_ratio))
        train_val_inds = [[inds[0:line], inds[line:]] for _ in np.arange(0, n_parallel_models)]
    for train_inds, val_inds in train_val_inds:
        x_train += [input_data[train_inds, ...]]
        if val_ratio > 0:
            x_val += [input_data[val_inds, ...]]
        y_train += [output_data[train_inds, ...]]
        if val_ratio > 0:
            y_val += [output_data[val_inds, ...]]

    return x_train, x_val, y_train, y_val, train_val_inds


def update_specs(data_specs, input_data, output_data, cat_dictionary):
    """ Update specs given data specs and input and output data.

        Parameters:
            data_specs (dict): Data specifications.
            input_data (pd.DataFrame): Input data.
            output_data (pd.DataFrame): Output data.
            cat_dictionary (pd.DataFrame): Categorical dictionary.
    """
    for specs_name, prep_data in zip(data_specs.keys(), [input_data, output_data]):
        specs = data_specs[specs_name]
        for feature_name, feature_specs in specs.items():
            dim = prep_data[feature_name][0].shape[-1] if not feature_specs['type'] == 'cat' \
                else len(cat_dictionary[feature_name + '_dictionary'][0])
            dim = 1 if dim == 2 and feature_specs['type'] == 'cat' and specs_name != 'output_specs' else dim
            feature_specs.update({'dim': dim})
