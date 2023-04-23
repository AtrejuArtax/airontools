from __future__ import annotations

import warnings

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as k_bcknd
from tensorflow.python.keras.mixed_precision.policy import Policy, set_global_policy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import Regularizer, l1, l1_l2, l2


def get_latent_model(model: Model, layer_name: str):
    try:
        return Model(
            inputs=model.inputs,
            outputs=model.get_layer(layer_name).output,
            name=layer_name + "_model",
        )
    except ValueError:
        outputs = model.inputs
        layer_found = False
        for layer in model.layers:
            outputs = layer(outputs)
            if layer.name == layer_name:
                layer_found = True
                break
        if layer_found:
            return Model(
                inputs=model.inputs,
                outputs=outputs,
                name=layer_name + "_model",
            )
        else:
            warnings.warn("could not find the layer")


def set_precision(precision: str) -> None:
    if "float16" in precision:
        if precision == "mixed_float16":
            policy = Policy("mixed_float16")
            set_global_policy(policy)
        else:
            tf.keras.backend.set_floatx("float16")


def to_time_series(tensor: tf.Tensor) -> tf.Tensor:
    return k_bcknd.expand_dims(tensor, axis=2)


def get_layer_units(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    min_hidden_units=2,
) -> list:
    units = [
        max(int(units), min_hidden_units)
        for units in np.linspace(input_dim, output_dim, n_layers + 1)
    ]
    units[0], units[-1] = input_dim, output_dim
    return units


def rm_redundant(values: list, value: int) -> list:
    taken = False
    values_ = []
    for n in values:
        if n != value:
            values_ += [n]
        elif not taken:
            values_ += [n]
            taken = True
    return values_


def get_regularizer(l1_value: float = None, l2_value: float = None) -> Regularizer:
    if l1_value and l2_value:
        return l1_l2(l1=l1_value, l2=l2_value)
    elif l1_value:
        return l1(l1_value)
    elif l2_value:
        return l2(l2_value)
