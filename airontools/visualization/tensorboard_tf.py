import os

import tensorflow as tf
from tensorboard.plugins import projector

from airontools.tools import path_management


def save_representations(representations, path, representations_name='representations', metadata=None):
    """ Save representations (or embeddings).

        Parameters:
            representations (array): Representations to be saved.
            path (str): Path to save the representations.
            representations_name (str): Embeddings names.
            metadata (list(array), array): Metadata (a list of arrays or an array).
    """

    # Path management
    path_management(path)

    # Save metadata
    metadata_file_name = os.path.join(path, 'metadata.tsv')
    if metadata is not None:
        metadata_list = metadata if isinstance(metadata, list) else [metadata]
        with open(metadata_file_name, 'w') as f:
            for i in range(len(metadata_list[0])):
                meta_line = []
                for metadata_ in metadata_list:
                    meta_line += [str(metadata_[i])]
                f.write('\t'.join(meta_line) + '\n')

    # Save representations
    representations_var = tf.Variable(representations, name=representations_name)
    checkpoint = tf.train.Checkpoint(embedding=representations_var)
    checkpoint.save(os.path.join(path, representations_name + '.ckpt'))

    # Set up config and representations
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
    embedding.metadata_path = metadata_file_name
    projector.visualize_embeddings(path, config)
