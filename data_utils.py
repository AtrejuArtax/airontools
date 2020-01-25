from __future__ import print_function
import numpy as np
from sklearn.model_selection import train_test_split
import random
import h5py
from tensorflow.keras import utils
import os
random.seed(0)
np.random.seed(0)
FLAGS = None


__author__ = 'claudi'

# Load and stack datasets
count = 0
X = []
Y = []
data_folder = '../deep_models_data/experiment_1/'
for file in os.listdir(data_folder):
    if file.endswith('_sim_data.h5'):
        print(file)
        if count == 0:
            X = np.array(h5py.File(data_folder + file, 'r')['motor']['block0_values'])[:20200, :]
            Y = np.array(h5py.File(data_folder + file, 'r')['somato']['block0_values'])[:20200]
        else:
            X = np.vstack((X, np.array(h5py.File(data_folder + file, 'r')['motor']['block0_values'])[:20200, :]))
            Y = np.vstack((Y, np.array(h5py.File(data_folder + file, 'r')['somato']['block0_values'])[:20200]))
        count += 1
Y = utils.to_categorical(Y, num_classes=2)

# Test and training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.16666666666666666666, random_state=0)
