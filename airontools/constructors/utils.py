import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k_bcknd
from tensorflow.keras import regularizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model


def get_latent_model(model: Model, layer_name: str):
    # ToDo: Make it for models with a non-trivial architecture
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
            name=layer_name + '_model'
        )
    else:
        warnings.warn('could not find the layer')


def set_precision(precision: float):
    if 'float16' in precision:
        if precision == 'mixed_float16':
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        else:
            tf.keras.backend.set_floatx('float16')


def to_time_series(tensor: tf.Tensor) -> tf.Tensor:
    return k_bcknd.expand_dims(tensor, axis=2)


def get_layer_units(input_dim: int, output_dim: int, n_layers: int, min_hidden_units=2) -> list:
    units = [max(int(units), min_hidden_units) for units in np.linspace(input_dim, output_dim, n_layers + 1)]
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


def regularizer(l1=None, l2=None) -> regularizers.Regularizer:
    if l1 and l2:
        regularizer = regularizers.l1_l2(l1=l1, l2=l2)
    elif l1:
        regularizer = regularizers.l1(l1)
    elif l2:
        regularizer = regularizers.l2(l2)
    else:
        regularizer = None
    return regularizer
