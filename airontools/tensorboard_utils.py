import os
from typing import List, Union

import tensorflow as tf
from numpy.typing import NDArray
from tensorboard.plugins import projector

from airontools.path_utils import path_management


def save_representations(
    representations: NDArray[float],
    path: str,
    representations_name: str = "representations",
    metadata: Union[
        List[NDArray[Union[float, str]]], NDArray[Union[float, str]]
    ] = None,
) -> None:
    """Save representations (or embeddings).

    Parameters:
        representations (NDArray[float]): Representations to be saved.
        path (str): Path to save the representations.
        representations_name (str): Embeddings name.
        metadata (Union[List[NDArray[Union[float, str]]], NDArray[Union[float, str]]]): Metadata (a list of arrays or an array).
    """

    # Path management
    path_management(path)

    # Save metadata
    metadata_file_name = __save_data(path=path, metadata=metadata)

    # Save representations
    __save_representations(
        path=path,
        representations=representations,
        representations_name=representations_name,
    )

    # Set up config and representations
    __set_conf(
        path=path,
        metadata_file_name=metadata_file_name,
    )


def __save_data(
    path: str,
    metadata: Union[
        List[NDArray[Union[float, str]]], NDArray[Union[float, str]]
    ] = None,
) -> str:
    metadata_file_name = os.path.join(path, "metadata.tsv")
    if metadata is not None:
        metadata_list = metadata if isinstance(metadata, list) else [metadata]
        with open(metadata_file_name, "w") as f:
            for i in range(len(metadata_list[0])):
                meta_line = []
                for metadata_ in metadata_list:
                    meta_line += [str(metadata_[i])]
                f.write("\t".join(meta_line) + "\n")
    return metadata_file_name


def __save_representations(
    path: str,
    representations: NDArray[float],
    representations_name: str = "representations",
) -> None:
    representations_var = tf.Variable(representations, name=representations_name)
    checkpoint = tf.train.Checkpoint(embedding=representations_var)
    checkpoint.save(os.path.join(path, representations_name + ".ckpt"))


def __set_conf(
    path: str,
    metadata_file_name: str,
) -> None:
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = metadata_file_name
    projector.visualize_embeddings(path, config)
