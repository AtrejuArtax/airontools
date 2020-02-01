import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


__author__ = 'claudi'


class PreProcessing(object):

    def __init__(self, num_features=None, cat_features=None, text_features=None, language='english'):
        self.num_features = num_features
        self.cat_features = cat_features
        self.text_features = text_features
        self.prep_features = []
        self.__mean = None
        self.__std = None
        self.__cat_encoder = None
        self.__vectorizer = None
        self.language = language

    def fit(self, x):

        # Numerital features: mean and std
        if self.num_features is not None and len(self.num_features) > 0:
            sub_x = x.loc[:, self.num_features].values
            mean, std = [], []
            for i in np.arange(0, sub_x.shape[1]):
                updated_array = sub_x[:, i][~np.isnan(sub_x[:, i])]
                mean += [np.mean(updated_array, axis=0)]
                std += [np.std(updated_array, axis=0)]
            self.__mean, self.__std = np.array(mean), np.array(std)
            self.prep_features += self.num_features

        # Categorical features: encoder
        if self.cat_features is not None and len(self.cat_features) > 0:
            sub_x = x.loc[:, self.cat_features].values.astype(str)
            sub_x = np.where(sub_x == '0', 'No', sub_x)
            sub_x = np.where(sub_x == '1', 'Yes', sub_x)
            sub_x = np.nan_to_num(sub_x)
            self.__cat_encoder = OneHotEncoder(handle_unknown='ignore')
            self.__cat_encoder.fit(sub_x)
            new_categories = []
            for array, cat_feature in zip(self.__cat_encoder.categories_, self.cat_features):
                if 'nan' in array:
                    array = array.tolist()
                    array.remove('nan')
                    array = np.array(array)
                new_categories += [array]
                self.prep_features += [cat_feature + '_' + category for category in array.tolist()]
            self.__cat_encoder.categories_ = new_categories

        # Text features: vectorizer
        if self.text_features is not None and len(self.text_features) > 0:
            sub_x = x.loc[:, self.text_features].values
            self.__vectorizer = TfidfVectorizer(stop_words=self.language, max_features=5000)
            self.__vectorizer.fit(sub_x)

    def transform(self, x):

        x_arrays = []

        # Numerical features
        if self.num_features is not None and len(self.num_features) > 0:
            x_arrays += [np.nan_to_num(x.loc[:, self.num_features].values)]
            x_arrays[-1] = (x_arrays[-1] - self.__mean) / self.__std

        # Categorical features
        if self.cat_features is not None and len(self.cat_features) > 0:
            x_arrays += [x.loc[:, self.cat_features].values.astype(str)]
            x_arrays[-1] = np.where(x_arrays[-1] == '0', 'No', x_arrays[-1])
            x_arrays[-1] = np.where(x_arrays[-1] == '1', 'Yes', x_arrays[-1])
            x_arrays[-1] = np.nan_to_num(x_arrays[-1])
            x_arrays[-1] = self.__cat_encoder.transform(x_arrays[-1]).toarray()

        # Text features
        if self.text_features is not None and len(self.text_features) > 0:
            x_arrays += [x.loc[:, self.text_features].values.astype(str)]
            x_arrays[-1] = self.__vectorizer.transform(x_arrays[-1])

        return pd.DataFrame(data=np.concatenate(tuple(x_arrays), axis=1), columns=self.prep_features)
