import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import KFold


def sub_sample(data, n):
    data_ = data.copy()
    data_.index = np.arange(data_.shape[0])
    return data.loc[:n-1, data_.columns]


def array_to_list(input_data, output_data, n_parallel_models, data_specs, do_kfolds=False, val_ratio=0.2,
                      shuffle=True, seed_val=0):
    """From array to list of numpy.

        Parameters:
            input_data (np.ndarray): Input data.
            output_data (np.ndarray): Output data.
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
        n_train = min([data[0].shape[0] for data in kf.split(range(input_data.shape[0]))])
        n_val = min([data[1].shape[0] for data in kf.split(range(input_data.shape[0]))])
        train_val_inds = [[train_inds[:n_train, ...], val_inds[:n_val, ...]]
                          for train_inds, val_inds in kf.split(range(input_data.shape[0]))]
    else:
        inds = np.arange(0, input_data.shape[0])
        if shuffle:
            import random
            from random import seed
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



class PreProcessing(object):

    def __init__(self, num_features=None, cat_features=None, text_features=None, language='english'):
        self.num_features = num_features
        self.cat_features = cat_features
        self.text_features = text_features
        self.prep_features = []
        self.mean = None
        self.std = None
        self.cat_encoder = None
        self.vectorizer = {}
        self.language = language

    def fit(self, x):

        # Numerical features: mean and std
        if self.num_features is not None and len(self.num_features) > 0:
            sub_x = x.loc[:, self.num_features].values
            mean, std = [], []
            for i in np.arange(0, sub_x.shape[1]):
                updated_array = sub_x[:, i][~np.isnan(sub_x[:, i])]
                mean += [np.mean(updated_array, axis=0)]
                std += [np.std(updated_array, axis=0)]
            self.mean, self.std = np.array(mean), np.array(std)
            self.prep_features += self.num_features

        # Categorical features: encoder
        if self.cat_features is not None and len(self.cat_features) > 0:
            sub_x = x.loc[:, self.cat_features].values.astype(str)
            sub_x = np.where(sub_x == '0', 'No', sub_x)
            sub_x = np.where(sub_x == '1', 'Yes', sub_x)
            sub_x = np.nan_to_num(sub_x)
            self.cat_encoder = OneHotEncoder(handle_unknown='ignore')
            self.cat_encoder.fit(sub_x)
            new_categories = []
            for array, cat_feature in zip(self.cat_encoder.categories_, self.cat_features):
                if 'nan' in array:
                    array = array.tolist()
                    array.remove('nan')
                    array = np.array(array)
                new_categories += [array]
                self.prep_features += [cat_feature + '_' + category for category in array.tolist()]
            self.cat_encoder.categories_ = new_categories

        # Text features: vectorizer
        if self.text_features is not None and len(self.text_features) > 0:
            for text_feature in self.text_features:
                sub_x = x[text_feature]
                sub_x = sub_x.apply(preprocessor)
                self.vectorizer.update({text_feature: TfidfVectorizer(stop_words=self.language, max_features=5000)})
                self.vectorizer[text_feature].fit(sub_x)
                self.prep_features += [text_feature]

    def transform(self, x):

        x_arrays = []

        # Numerical features
        if self.num_features is not None and len(self.num_features) > 0:
            sub_x = x.loc[:, self.num_features]
            x_arrays += [np.nan_to_num(sub_x.values)]
            x_arrays[-1] = (x_arrays[-1] - self.mean) / self.std

        # Categorical featuresprep_tools_general
        if self.cat_features is not None and len(self.cat_features) > 0:
            sub_x = x.loc[:, self.cat_features]
            x_arrays += [sub_x.values.astype(str)]
            x_arrays[-1] = np.where(x_arrays[-1] == '0', 'No', x_arrays[-1])
            x_arrays[-1] = np.where(x_arrays[-1] == '1', 'Yes', x_arrays[-1])
            x_arrays[-1] = np.nan_to_num(x_arrays[-1])
            x_arrays[-1] = self.cat_encoder.transform(x_arrays[-1]).toarray()

        # Text features
        if self.text_features is not None and len(self.text_features) > 0:
            for text_feature in self.text_features:
                sub_x = x[text_feature]
                sub_x = sub_x.apply(preprocessor)
                x_arrays += [self.vectorizer[text_feature].transform(sub_x)]
        if len(x_arrays) == 1:
            x_arrays = x_arrays[0]
        else:
            x_arrays = np.concatenate(tuple(x_arrays), axis=1)
        return x_arrays


def tokenize_it(data, tokenizer, sequential=False):
    t_data = tokenizer.texts_to_matrix(data.to_list(), mode='binary')[:, 1:]
    return [[t_data[i].tolist()] for i in np.arange(0, t_data.shape[0])]


def preprocessor(sentence):
    sentence = sentence.strip().lower()
    sentence = re.sub(r"\d+", "", sentence)
    sentence = sentence.translate(sentence.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    sentence = " ".join([w for w in nltk.word_tokenize(sentence) if len(w) > 1])
    return sentence


def get_processed_data_(files_for, file_names, target_column, skip_columns, features, mult=1):
    X, Y, prep, target_prep = {}, {}, None, None
    for file_name, file_for in zip(file_names, files_for):
        # Load
        dataset = pd.read_csv(file_name, sep="\t", encoding="utf-8").reset_index(drop=True)
        n = dataset.shape[0] if file_for != 'train' and mult != 1 else dataset.shape[0] - dataset.shape[0] % mult
        dataset = dataset.sample(n=n)
        X.update({file_for: dataset.loc[:, [header for header in dataset.keys()
                                            if header not in [target_column] + skip_columns]]})
        Y.update({file_for: dataset.loc[:, [target_column]]})
        # Preprocess/Transform
        if file_for == 'train':
            prep = PreProcessing(text_features=features['text'], language='english')
            prep.fit(X[file_for])
            target_prep = PreProcessing(cat_features=[target_column])
            target_prep.fit(Y[file_for])
        X[file_for] = prep.transform(X[file_for])
        Y['raw_' + file_for] = Y[file_for].copy()
        Y['raw_' + file_for] = Y[file_for].copy()
        Y[file_for] = target_prep.transform(Y[file_for])
    return X, Y, prep, target_prep

