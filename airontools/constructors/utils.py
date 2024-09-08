import warnings
from typing import Optional

import keras


def get_latent_model(
    model: keras.models.Model, layer_name: str
) -> Optional[keras.models.Model]:
    """Gets latent model.

    Parameters:
        model (keras.models.Model): Model.
        layer_name (str): Layer name from which to represent the data.

    Returns:
        A keras.models.Model.
    """
    try:
        return keras.models.Model(
            inputs=model._model.inputs,
            outputs=model._model.get_layer(layer_name).outputs,
            name=layer_name + "_model",
        )
    except ValueError:
        outputs = model._model.inputs
        layer_found = False
        for layer in model._model.layers[1:]:
            outputs = layer(outputs)
            if layer.name == layer_name:
                layer_found = True
                break
        if layer_found:
            return keras.models.Model(
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
            policy = keras.mixed_precision.Policy("mixed_float16")
            keras.mixed_precision.set_global_policy(policy)
        else:
            keras.backend.set_floatx("float16")


def get_regularizer(
    l1_value: Optional[float] = None, l2_value: Optional[float] = None
) -> keras.regularizers.Regularizer:
    """Gets a regularizer.

    Parameters:
        l1_value (float): L1 (Lasso) regularization value.
        l2_value (float): L2 (Ridge) regularization value.

    Returns:
        A keras.regularizers.Regularizer.
    """
    if l1_value and l2_value:
        return keras.regularizers.l1_l2(l1=l1_value, l2=l2_value)
    elif l1_value:
        return keras.regularizers.l1(l1=l1_value)
    elif l2_value:
        return keras.regularizers.l2(l2=l2_value)
