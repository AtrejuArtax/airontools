import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import nltk


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

        # Numerital features: mean and std
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

        # Categorical features
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


def preprocessor(sentence):
    sentence = sentence.strip().lower()
    sentence = re.sub(r"\d+", "", sentence)
    sentence = sentence.translate(sentence.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    sentence = " ".join([w for w in nltk.word_tokenize(sentence) if len(w) > 1])
    return sentence
