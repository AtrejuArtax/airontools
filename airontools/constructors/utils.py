import warnings
from typing import Optional

import tensorflow as tf


def get_latent_model(
    model: tf.keras.models.Model, layer_name: str
) -> Optional[tf.keras.models.Model]:
    """Gets latent model.

    Parameters:
        model (tf.keras.models.Model): Model.
        layer_name (str): Layer name from which to represent the data.

    Returns:
        A tf.keras.models.Model.
    """
    try:
        return tf.keras.models.Model(
            inputs=model._model.inputs,
            outputs=model._model.get_layer(layer_name).outputs,
            name=layer_name + "_model",
        )
    except ValueError:
        outputs = model._model.inputs
        layer_found = False
        for layer in model._model.layers:
            outputs = layer(outputs)
            if layer.name == layer_name:
                layer_found = True
                break
        if layer_found:
            return tf.keras.models.Model(
                inputs=model._model.inputs,
                outputs=outputs,
                name=layer_name + "_model",
            )
        else:
            warnings.warn("could not find the layer")


def set_precision(precision: str) -> None:
    """Sets variables precision.

    Parameters:
        precision (str): Precision of the variables.
    """
    if "float16" in precision:
        if precision == "mixed_float16":
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
        else:
            tf.keras.backend.set_floatx("float16")


def get_regularizer(
    l1_value: float = None, l2_value: float = None
) -> tf.keras.regularizers.Regularizer:
    """Gets a regularizer.

    Parameters:
        l1_value (float): L1 (Lasso) regularization value.
        l2_value (float): L2 (Ridge) regularization value.

    Returns:
        A tf.keras.regularizers.Regularizer.
    """
    if l1_value and l2_value:
        return tf.keras.regularizers.l1_l2(l1=l1_value, l2=l2_value)
    elif l1_value:
        return tf.keras.regularizers.l1(l1=l1_value)
    elif l2_value:
        return tf.keras.regularizers.l2(l2=l2_value)
