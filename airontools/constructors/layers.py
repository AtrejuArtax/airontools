import warnings
from typing import Tuple, Union

import numpy as np
import tensorflow as tf

from airontools.constructors.utils import get_regularizer


def layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    units: int,
    name: str = "layer",
    name_ext: str = "",
    num_heads: int = 0,
    key_dim: int = 0,
    value_dim: int = 0,
    activation: Union[str, tf.keras.layers.Activation] = "linear",
    use_bias: bool = True,
    sequential: bool = False,
    bidirectional: bool = False,
    return_sequences: bool = False,
    filters: int = 0,
    kernel_size: Union[int, Tuple[int]] = 0,
    padding: str = "valid",
    pooling: Union[str, tf.keras.layers.Layer] = None,
    pool_size: Union[int, Tuple[int]] = 1,
    conv_transpose: bool = False,
    strides: Union[int, Tuple[int]] = 1,
    sequential_axis: int = 1,
    kernel_regularizer_l1: float = 0.001,
    kernel_regularizer_l2: float = 0.001,
    bias_regularizer_l1: float = 0.001,
    bias_regularizer_l2: float = 0.001,
    dropout_rate: float = 0.0,
    bn: bool = False,
) -> tf.Tensor:
    """It builds a custom layer. For now only 2D convolutions
    are supported for input of rank 4.

    Parameters:
        x (tf.Tensor, tf.keras.layers.Layer): Input layer or tensor.
        units (int): Number of units for the dense layer. If a value is given, a dense layer will be added
        automatically if not sequential, else a sequential model. Useful to force an output dimensionality of the
        custom layer when using convolutional layers.
        name (str): Name of the custom layer.
        name_ext (str): Extension name for the custom layer that will be at the end of it.
        num_heads (int): Number of heads for the multi-head attention layer.
        key_dim (int): Key dimensionality for the multi-head attention layer, if None then the number of units is
        used instead.
        value_dim (int): Value dimensionality for the multi-head attention layer, if None then key_dim is used
        instead.
        activation (str, tf.keras.layers.Activation): The activation function of the output of the last hidden layer.
        use_bias (bool): Whether to sue bias or not.
        sequential (bool): Whether to consider a sequential custom layer or not. Sequential and self-attention
        (num_heads > 0) are not compatible.
        bidirectional (bool): Whether to consider bidirectional case or not (only active if sequential).
        names are not repeated.
        return_sequences (bool): Whether to return sequences or not (only active if sequential).
        filters (int): Number of filters to be used. If a value is given, a convolutional layer will be
        automatically added.
        kernel_size (int, Tuple[int]): Kernel size for the convolutional layer.
        conv_transpose (bool): Whether to use a transpose conv layer or not (only active if filters and
        kernel_size are set).
        strides (int, Tuple[int]): Strides for the conv layer (only active if filters and
        kernel_size are set).
        padding (str): Padding to be applied (only active if filters and
        kernel_size are set).
        pooling (str, tf.keras.layers.Layer): Pooling type.
        pool_size (int, Tuple[int]): Pooling size.
        sequential_axis (int): The axis that defines the sequence. For sequential models is normally 1. For
        self-attention (num_heads > 0) and image-like inputs, the sequential axis is the channel axis (3 for 2D
        images and 4 for 3D images).
        kernel_regularizer_l1 (float): Kernel regularization using l1 penalization (Lasso).
        kernel_regularizer_l2 (float): Kernel regularization using l2 penalization (Ridge).
        bias_regularizer_l1 (float): Bias regularization using l1 penalization (Lasso).
        bias_regularizer_l2 (float): Bias regularization using l2 penalization (Ridge).
        dropout_rate (float): Dropout rate.
        bn (bool): If set, a batch normalization layer will be added right before the output activation function.

    Returns:
        x (tf.keras.layers.Layer): A keras layer.
    """

    if num_heads > 0 and units is None and key_dim == 0:
        warnings.warn(
            "in order to use a multi-head attention layer either units or key_dim needs to be set",
        )

    # Dropout
    if dropout_rate != 0:
        x = dropout_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            dropout_rate=dropout_rate,
        )

    # Convolution
    conv_condition = all(
        [conv_param != 0 for conv_param in [filters, kernel_size]],
    )
    if conv_condition:
        pooling_dim = _get_pooling_dim(x)
        if isinstance(strides, int) and pooling_dim > 0:
            _strides = tuple([strides] * pooling_dim)
        else:
            _strides = strides
        conv_kwargs = dict(
            use_bias=use_bias,
            filters=filters,
            kernel_size=kernel_size,
            strides=_strides,
            padding=padding,
            kernel_regularizer=get_regularizer(
                kernel_regularizer_l1,
                kernel_regularizer_l2,
            ),
            bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
        )
        x = convolutional_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            conv_transpose=conv_transpose,
            **conv_kwargs,
        )

    # Pooling
    if pooling is not None:
        pooling_dim = _get_pooling_dim(x)
        if isinstance(pool_size, int) and pooling_dim > 0:
            _pool_size = tuple([pool_size] * pooling_dim)
        else:
            _pool_size = pool_size
        pooling_kwargs = dict(
            pool_size=_pool_size,
            strides=strides,
            padding=padding,
        )
        x = pooling_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            conv_transpose=conv_transpose,
            pooling=pooling,
            **pooling_kwargs,
        )

    # Multi-Head Attention
    if num_heads > 0:
        _key_dim = key_dim if key_dim > 0 else units
        _value_dim = value_dim if value_dim > 0 else _key_dim
        multi_head_attention_kwargs = dict(
            num_heads=num_heads,
            key_dim=_key_dim,
            value_dim=_value_dim,
            use_bias=use_bias,
            kernel_regularizer=get_regularizer(
                kernel_regularizer_l1,
                kernel_regularizer_l2,
            ),
            bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
        )
        x = self_attention_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            sequential_axis=sequential_axis,
            **multi_head_attention_kwargs,
        )

    # Sequential
    elif sequential:
        seq_kwargs = dict(
            units=units,
            use_bias=use_bias,
            kernel_regularizer=get_regularizer(
                kernel_regularizer_l1,
                kernel_regularizer_l2,
            ),
            bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
            return_sequences=return_sequences,
            activation="linear",
        )
        x = sequential_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            bidirectional=bidirectional,
            sequential_axis=sequential_axis,
            **seq_kwargs,
        )

    # Dense
    if units > 0:
        dense_kwargs = dict(
            units=units,
            use_bias=use_bias,
            kernel_regularizer=get_regularizer(
                kernel_regularizer_l1,
                kernel_regularizer_l2,
            ),
            bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
        )
        x = dense_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            **dense_kwargs,
        )

    # Batch Normalization
    if bn:
        bn_kwargs = dict(
            beta_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
            gamma_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
        )
        x = bn_layer_constructor(x, name=name, name_ext=name_ext, **bn_kwargs)

    # Activation
    activation_kwargs = dict(
        alpha_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
    )
    x = activation_layer_constructor(
        x,
        name=name,
        name_ext=name_ext,
        activation=activation,
        **activation_kwargs,
    )

    return x


def dropout_layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    dropout_rate: float,
    name: str = "dropout",
    name_ext: str = "",
) -> tf.keras.layers.Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        output_reshape = input_shape
        x = tf.keras.layers.Flatten(
            name="_".join([name, "pre", "dropout", "flatten", name_ext])
        )(x)
    x = CustomDropout(name="_".join([name, "dropout", name_ext]), rate=dropout_rate)(x)
    if output_reshape is not None:
        x = tf.keras.layers.Reshape(
            name="_".join([name, "post", "dropout", "reshape", name_ext]),
            target_shape=output_reshape[1:],
        )(x)
    return x


def convolutional_layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    name: str = "convolution",
    name_ext: str = "",
    conv_transpose: bool = False,
    **kwargs,
) -> tf.keras.layers.Layer:
    assert len(x.shape) <= 5, "x layer shape should have 5 or less dimensions"
    conv_dim = len(x.shape) - 2
    conv_type = "transpose" if conv_transpose else ""
    conv_name = "conv" + str(conv_dim) + "d" + conv_type.capitalize()
    layer_name = "_".join([name, conv_name, name_ext])
    if conv_dim == 1:
        conv_layer = tf.keras.layers.Conv1D
    elif conv_dim == 2:
        conv_layer = tf.keras.layers.Conv2D
    else:
        conv_layer = tf.keras.layers.Conv3D
    x = conv_layer(
        name=layer_name,
        **kwargs,
    )(x)
    return x


def pooling_layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    name: str = "pooling",
    name_ext: str = "",
    pooling: Union[str, tf.keras.layers.Layer] = "max",
    **kwargs,
) -> tf.keras.layers.Layer:
    if isinstance(pooling, str):
        pooling_name = pooling.capitalize() + "Pooling" + str(_get_pooling_dim(x)) + "D"
        pooling_layer = globals()[pooling_name]
    else:
        pooling_name = [k for k, v in locals().iteritems() if v == pooling][0]
        pooling_layer = pooling
    x = pooling_layer(name="_".join([name, pooling_name.lower(), name_ext]))(x)
    return x


def _get_pooling_dim(x: tf.keras.layers.Layer) -> int:
    return len(x.shape) - 2


def self_attention_layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    name: str = "self_attention",
    name_ext: str = "",
    sequential_axis: int = 1,
    **kwargs,
) -> tf.keras.layers.Layer:
    x = sequential_permutation(
        x=x,
        name=name,
        name_ext=name_ext,
        sequential_axis=sequential_axis,
    )
    x = tf.keras.layers.MultiHeadAttention(
        name="_".join([name, "multi_head_attention", name_ext]),
        **kwargs,
    )(
        x,
        x,
        x,
        use_causal_mask=True,
    )
    return x


def sequential_layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    name: str = "sequential",
    name_ext: str = "",
    bidirectional: bool = False,
    sequential_axis: int = 1,
    **kwargs,
) -> tf.keras.layers.Layer:
    x = sequential_permutation(
        x=x,
        name=name,
        name_ext=name_ext,
        sequential_axis=sequential_axis,
    )
    if bidirectional:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(name="_".join([name, "gru", name_ext]), **kwargs)
        )(x)
    else:
        x = tf.keras.layers.GRU(name="_".join([name, "gru", name_ext]), **kwargs)(x)
    return x


def dense_layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    name: str = "dense",
    name_ext: str = "",
    **kwargs,
) -> tf.keras.layers.Layer:
    if not len(x.shape[1:]) == 1:
        x = tf.keras.layers.Flatten(
            name="_".join([name, "pre", "dense", "flatten", name_ext])
        )(x)
    x = tf.keras.layers.Dense(name="_".join([name, "dense", name_ext]), **kwargs)(x)
    return x


def bn_layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer], name: str, name_ext: str, **kwargs
) -> tf.keras.layers.Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        output_reshape = input_shape
        x = tf.keras.layers.Flatten(
            name="_".join([name, "pre", "bn", "flatten", name_ext])
        )(x)
    x = tf.keras.layers.BatchNormalization(
        name="_".join([name, "batch_normalization", name_ext]),
        **kwargs,
    )(x)
    if output_reshape is not None:
        x = tf.keras.layers.Reshape(
            name="_".join([name, "post", "bn", "reshape", name_ext]),
            target_shape=output_reshape[1:],
        )(x)
    return x


def activation_layer_constructor(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    name: str = "activation",
    name_ext: str = "",
    activation: str = "linear",
    **reg_kwargs,
) -> tf.keras.layers.Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        output_reshape = input_shape
        x = tf.keras.layers.Flatten(
            name="_".join([name, "pre", "activation", "flatten", name_ext])
        )(x)
    activation_name = "_".join(
        [name, activation if isinstance(activation, str) else "activation", name_ext],
    )
    if activation == "leakyrelu":
        x = tf.keras.layers.LeakyReLU(name=activation_name)(x)
    elif activation == "prelu":
        x = tf.keras.layers.PReLU(name=activation_name, **reg_kwargs)(x)
    elif activation == "softmax":
        x = tf.keras.layers.Softmax(name=activation_name, dtype="float32")(x)
    else:
        x = tf.keras.layers.Activation(name=activation_name, activation=activation)(x)
    if output_reshape is not None:
        x = tf.keras.layers.Reshape(
            name="_".join([name, "post", "activation", "reshape", name_ext]),
            target_shape=output_reshape[1:],
        )(x)
    return x


def sequential_permutation(
    x: Union[tf.Tensor, tf.keras.layers.Layer],
    name: str = "permutation",
    name_ext: str = "",
    sequential_axis: int = 1,
) -> tf.keras.layers.Layer:
    input_shape = x.shape
    sequential_axis_ = list(range(len(input_shape)))[sequential_axis]
    if sequential_axis_ != 1:
        permutation = tuple(
            [sequential_axis_]
            + [i for i in range(1, len(input_shape[1:])) if i != sequential_axis_],
        )
        x = tf.keras.layers.Permute(
            name="_".join([name, "permutation", name_ext]), dims=permutation
        )(x)
    else:
        permutation = list(range(1, len(input_shape)))
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        permutation_dimensions = [input_shape[i] for i in permutation[1:]]
        x = tf.keras.layers.Reshape(
            name="_".join([name, "reshape", name_ext]),
            target_shape=(
                input_shape[permutation[0]],
                np.prod(permutation_dimensions),
            ),
        )(x)
    return x


def identity(x: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """Identity layer."""
    return x


class CustomDropout(tf.keras.layers.Dropout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rate = tf.Variable(
            self.rate,
            trainable=False,
            name="_".join([self.name, "rate"]),
            dtype=self.dtype,
        )
