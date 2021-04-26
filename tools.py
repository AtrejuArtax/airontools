import os
import numpy as np
import shutil


def path_preparation(paths, new_preprocessing=True, new_exploration=True):
    """Path preparation.

        Parameters:
            paths (list): Paths to prepare.
            new_preprocessing (bool): Whether it requires new pre-processing or not.
            new_exploration (bool): Whether it requires new exploration or not.
    """
    for path in paths:
        make_dirtree(path)
        if (new_preprocessing and 'PrepDatasets' in path) or (new_exploration and 'O' in path):
            shutil.rmtree(path)
            make_dirtree(path)


def make_dirtree(path):
    """Make a tree of directories

        Parameters:
            path (str): Paths to prepare.
    """
    path_list = path.split('/')[1:-1]
    for i in np.arange(0, len(path_list)):
        path_ = '/' + '/'.join(path_list[0:i + 1]) + '/'
        if not os.path.isdir(path_):
            os.mkdir(path_)
