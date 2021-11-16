import os
import tempfile
import tensorflow as tf
from tensorboard.plugins import projector
from airontools.tools import path_management


def save_insights(embeddings, embeddings_names=None, metadata=None,
                  path=os.path.join(tempfile.gettempdir(), 'insights')):
    """ Save insights (embeddings or latent representations).

        Parameters:
            embeddings (list, array): Embeddings to be saved.
            embeddings_names (list, str): Embeddings names.
            metadata (list, array): Metadata.
            path (str): Path to save insights.
    """

    embeddings = embeddings if isinstance(embeddings, list) else [embeddings]
    embeddings_names = embeddings_names if embeddings_names else \
        list(['embeddings_' + str(i) for i in range(len(embeddings))])

    # Path management
    path_management(path)

    # Save metadata
    metadata_file_name = os.path.join(path, 'metadata.tsv')
    if metadata:
        with open(metadata_file_name, "w") as f:
            pass

    # Iterate over embeddings
    for embedding, embeddings_name in zip(embeddings, embeddings_names):

        # Save data
        checkpoint = tf.train.Checkpoint(embedding=embedding)
        checkpoint.save(os.path.join(path, embeddings_name + '.ckpt'))

        # Set up config and embeddings
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings_name
        embedding.metadata_path = metadata_file_name
        projector.visualize_embeddings(path, config)
