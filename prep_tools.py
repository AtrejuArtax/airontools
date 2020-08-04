import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.text import Tokenizer


def to_prep_data(start_week, end_week, t_plus, n_w_gone, test_from_w, static_data_file_name, sequential_data_file_name,
                 max_n_samples=None, nan_to=0, seq_length=None):
    """From raw to processed data.

        Parameters:
            start_week (int): Start week.
            end_week (int): End week.
            t_plus (int): Time plus value for prediction.
            n_w_gone (int): Number of weeks to consider gone.
            test_from_w (int): Test from specified week.
            static_data_file_name (str): Static data file name.
            sequential_data_file_name (str): Sequential data file name.
            max_n_samples (int): Maximum number of samples.

        Returns:
            2 pd.DataFrame and a list.
    """

    # Define features and others
    driver = 'week'
    identifier = 'courier'
    static_cat_features = ['feature_1']
    static_num_features = ['feature_2']
    static_features, static_name = static_cat_features + static_num_features, 'static'
    sequential_cat_features = ['feature_16']
    sequential_num_features = ['feature_{}'.format(i) for i in np.arange(1, 18) if i != 16]
    sequential_features, sequential_name = sequential_cat_features + sequential_num_features, 'sequential'

    # Load datasets
    file_names = [static_data_file_name, sequential_data_file_name]
    feature_names = [[identifier] + static_features, [identifier, driver] + sequential_features]
    datasets = []
    for file_name, feat_names in zip(file_names, feature_names):
        datasets += [pd.read_csv(file_name, usecols=feat_names)]
    static_data, sequential_data = datasets

    # Filter sequential data based on start end weeks
    sequential_data = sequential_data[sequential_data[driver] >= start_week]
    sequential_data = sequential_data[sequential_data[driver] <= end_week]

    # Replace nans by to_nan
    sequential_data = sequential_data.fillna(nan_to)
    static_data = static_data.fillna(nan_to)

    # Format
    sequential_data[sequential_cat_features] = sequential_data[sequential_cat_features].astype(str)

    # Reduce dataset to maximum amount of samples ( at least number of couriers)
    if max_n_samples is not None:
        couriers = sequential_data[identifier].unique()[:max_n_samples]
        sequential_data = sequential_data[sequential_data[identifier].isin(couriers)]
        static_data = static_data[static_data[identifier].isin(couriers)]

    # Sort sequential data based on week
    sequential_data = sequential_data.sort_values(driver, ascending=False)
    sequential_data.index = np.arange(0, sequential_data.shape[0])

    # Get preprocessing parameters from training data (everything before end_week - t_plus - n_w_gone)
    sequential_data_ = sequential_data[sequential_data[driver] <= test_from_w]
    static_data_ = static_data[static_data[identifier].isin(sequential_data_[identifier].unique())]
    static_mean, static_std = static_data_[static_num_features].mean(), static_data_[static_num_features].std()
    static_unique = [list(static_data_[cat_feature].unique()) for cat_feature in static_cat_features]
    static_tokenizers = [Tokenizer() for _ in static_cat_features]
    [t.fit_on_texts(dictionary) for t, dictionary in zip(static_tokenizers, static_unique)]
    seq_mean, seq_std = sequential_data_[sequential_num_features].mean(), sequential_data_[sequential_num_features].std()
    seq_unique = [list(sequential_data_[cat_feature].unique()) for cat_feature in sequential_cat_features]
    seq_tokenizers = [Tokenizer() for _ in sequential_cat_features]
    [t.fit_on_texts(dictionary) for t, dictionary in zip(seq_tokenizers, seq_unique)]
    del sequential_data_, static_data_

    # Generate input and output data
    train_i_data, train_o_data = [], []
    test_i_data, test_o_data = [], []
    for end_week_ in np.arange(start_week, end_week + 1)[::-1]:
        filt_seq_data = sequential_data[sequential_data[driver] <= end_week_]
        groups = filt_seq_data.groupby(identifier)
        o_data_ = test_o_data if test_from_w <= end_week_ - t_plus - n_w_gone else train_o_data
        i_data_ = test_i_data if test_from_w <= end_week_ - t_plus - n_w_gone else train_i_data
        for key in groups.groups:
            future_weeks = [i for i in np.arange(end_week_ - n_w_gone + 1, end_week_ + 1)]
            past_weeks = [i for i in np.arange(start_week, end_week_ - t_plus - n_w_gone + 1)]
            if seq_length is not None:
                past_weeks = [i for i in np.arange(-seq_length, 0)] + past_weeks
                past_weeks = past_weeks[-seq_length:]
            group_ = groups.get_group(key)
            complete_seq_data = group_.copy()
            complete_seq_data = complete_seq_data[complete_seq_data[driver].isin(past_weeks)]
            for i in past_weeks:
                if not any(list(complete_seq_data[driver].isin([i]))):
                    empty_data = [key, i] + [nan_to] * len(sequential_features)
                    complete_seq_data = pd.concat([complete_seq_data,
                                                   pd.DataFrame(data=np.array(empty_data).reshape((1, len(empty_data))),
                                                                columns=complete_seq_data.columns)])
            complete_seq_data = complete_seq_data.sort_values(driver, ascending=False)[sequential_features]
            complete_seq_data.index = np.arange(0, complete_seq_data.shape[0])
            complete_seq_data[sequential_cat_features] = complete_seq_data[sequential_cat_features].astype(str)
            static_data_ = static_data[static_data[identifier] == key]

            # Preprocess
            prep_static_cat = pd.DataFrame(
                data=tokenize_it(static_data_[static_cat_features[0]], tokenizer=static_tokenizers[0]),
                columns=[static_cat_features[0]])
            prep_static_num = (static_data_[static_num_features] - static_mean) / static_std
            prep_sequential_cat = pd.DataFrame(
                data=tokenize_it(complete_seq_data[sequential_cat_features[0]], tokenizer=seq_tokenizers[0]),
                columns=[sequential_cat_features[0]])
            prep_sequential_num = (complete_seq_data[sequential_num_features] - seq_mean) / seq_std

            # Define samples
            i_data_ += [[prep_static_cat.values.tolist(),
                         prep_static_num.values.tolist(),
                         prep_sequential_cat.values.tolist(),
                         prep_sequential_num.values.tolist()]]
            o_data_ += [[[1, 0] if any(list(group_[driver].isin(future_weeks))) else [0, 1]]]

    # To data frames
    train_i_data = pd.DataFrame(data=train_i_data,
                                columns=['static_cat', 'static_num', 'sequential_cat', 'sequential_num'])
    train_o_data = pd.DataFrame(data=train_o_data, columns=['y'])
    test_i_data = pd.DataFrame(data=test_i_data,
                               columns=['static_cat', 'static_num', 'sequential_cat', 'sequential_num'])
    test_o_data = pd.DataFrame(data=test_o_data, columns=['y'])
    cat_dictionary = static_unique + seq_unique + [[0, 1]]
    cat_dictionary = pd.DataFrame(data=np.array(cat_dictionary).reshape((1, len(cat_dictionary))),
                         columns=['static_cat_dictionary', 'sequential_cat_dictionary', 'y_dictionary'])

    # Sample to get max number of samples
    if max_n_samples is not None:
        train_perc = 0.8
        train_n = int(max_n_samples * train_perc)
        test_n = max_n_samples - train_n
        train_i_data, train_o_data = sub_sample(train_i_data, train_n), sub_sample(train_o_data, train_n)
        test_i_data, test_o_data = sub_sample(test_i_data, test_n), sub_sample(test_o_data, test_n)

    return train_i_data, train_o_data, test_i_data, test_o_data, cat_dictionary


def sub_sample(data, n):
    data_ = data.copy()
    data_.index = np.arange(data_.shape[0])
    return data.loc[:n-1, data_.columns]


def dataframe_to_list(input_data, output_data, n_parallel_models, data_specs, do_kfolds=False, val_ratio=0.2,
                      shuffle=True, seed_val=0):
    """From dataframes to list of numpys.

        Parameters:
            input_data (pd.DataFrame): Input data.
            output_data (pd.DataFrame): Output data.
            n_parallel_models (int): Number of parallel models.
            data_specs (dict): Dataset specifications.
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
        train_val_inds = [[train_inds, val_inds] for train_inds, val_inds in kf.split(range(input_data.shape[0]))]
    else:
        inds = np.arange(0, input_data.shape[0])
        if shuffle:
            import random
            from random import seed
            random.shuffle(inds, random=seed(seed_val))
        line = int(len(inds) * (1 - val_ratio))
        train_val_inds = [[inds[0:line], inds[line:]] for _ in np.arange(0, n_parallel_models)]
    for train_inds, val_inds in train_val_inds:
        for input_name, input_specs in data_specs['input_specs'].items():
            if input_specs['sequential']:
                dim = input_specs['dim']
                x_train += [np.vstack(
                    [np.array(input_data.loc[train_inds, [input_name]].values[i][0]).reshape(
                        (1, input_specs['length'], dim)) for i in np.arange(0, len(train_inds))])]
                if val_ratio > 0:
                    x_val += [np.vstack(
                        [np.array(input_data.loc[val_inds, [input_name]].values[i][0]).reshape(
                            (1, input_specs['length'], dim)) for i in np.arange(0, len(val_inds))])]
            else:
                x_train += [np.vstack([input_data.loc[train_inds, [input_name]].values[i][0]
                                       for i in np.arange(0, len(train_inds))])]
                if val_ratio > 0:
                    x_val += [np.vstack([input_data.loc[val_inds, [input_name]].values[i][0]
                                         for i in np.arange(0, len(val_inds))])]
        for output_name in data_specs['output_specs'].keys():
            y_train += [np.vstack([output_data.loc[train_inds, [output_name]].values[i][0]
                                   for i in np.arange(0, len(train_inds))])]
            if val_ratio > 0:
                y_val += [np.vstack([output_data.loc[val_inds, [output_name]].values[i][0]
                                     for i in np.arange(0, len(val_inds))])]

    return x_train, x_val, y_train, y_val


def update_specs(data_specs, input_data, output_data, cat_dictionary):
    """Update specs given data specs and input and output data.

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


def tokenize_it(data, tokenizer, sequential=False):
    t_data = tokenizer.texts_to_matrix(data.to_list(), mode='binary')[:, 1:]
    return [[t_data[i].tolist()] for i in np.arange(0, t_data.shape[0])]
