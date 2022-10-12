import numpy as np
from numpy.random import normal

N_HALF_SAMPLES = 50
N_CLASSES = 2
IMG_DIM = 28
IMG_DATA = np.concatenate(
    [
        normal(loc=0.5, scale=1, size=(N_HALF_SAMPLES, IMG_DIM, IMG_DIM)),
        normal(loc=-0.5, scale=1, size=(N_HALF_SAMPLES, IMG_DIM, IMG_DIM)),
    ]
)
N_FEATURES = 10
TABULAR_DATA = np.concatenate(
    [
        normal(loc=0.5, scale=1, size=(N_HALF_SAMPLES, N_FEATURES)),
        normal(loc=-0.5, scale=1, size=(N_HALF_SAMPLES, N_FEATURES)),
    ]
)
TARGETS = np.concatenate(
    [
        [[1, 0]] * N_HALF_SAMPLES,
        [[0, 1]] * N_HALF_SAMPLES,
    ]
)
LOSS_TOLERANCE = 0.03
