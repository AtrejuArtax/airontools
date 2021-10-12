import tempfile
from airontools.tools import path_management
import os
import tensorflow as tf
from tensorboard.plugins import projector


def save_insight(embeddings, metadata=None, path=os.path.join(tempfile.gettempdir(), 'insights')):

    # Path management
    path_management(path)

    # Save metadata
    metadata_file_name = os.path.join(path, 'metadata.tsv')
    if metadata:
        with open(metadata_file_name, "w") as f:
            pass

    # Sve data
    checkpoint = tf.train.Checkpoint(embedding=embeddings)
    checkpoint.save(os.path.join(path, 'data.ckpt'))

    # Set up config and embeddings
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = metadata_file_name
    projector.visualize_embeddings(path, config)
