import warnings
from typing import Optional, Tuple, Union

import keras
import numpy as np
import tensorflow as tf

from airontools.constructors.utils import get_regularizer


def layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer],
    units: int,
    name: str = "layer",
    name_ext: Optional[str] = None,
    num_heads: int = 0,
    key_dim: int = 0,
    value_dim: int = 0,
    multi_head_attention_dropout_rate: float = 0.0,
    return_attention_scores: bool = False,
    use_causal_mask: bool = False,
    activation: Union[str, keras.layers.Activation] = "linear",
    sequential_activation: Union[str] = "tanh",
    use_bias: bool = True,
    sequential: bool = False,
    bidirectional: bool = False,
    return_sequences: bool = False,
    filters: int = 0,
    kernel_size: Union[int, Tuple[int]] = 0,
    padding: str = "valid",
    pooling: Optional[Union[str, keras.layers.Layer]] = None,
    pool_size: Union[int, Tuple[int]] = 1,
    conv_transpose: bool = False,
    strides: Union[int, Tuple[int]] = 1,
    sequential_axis: int = 1,
    kernel_regularizer_l1: float = 0.001,
    kernel_regularizer_l2: float = 0.001,
    bias_regularizer_l1: float = 0.001,
    bias_regularizer_l2: float = 0.001,
    dropout_rate: float = 0.0,
    normalization_type: Optional[str] = None,
) -> Union[keras.layers.Layer, Tuple[keras.layers.Layer, keras.layers.Layer]]:
    """It builds a custom layer. For now only 2D convolutions
    are supported for input of rank 4.

    Parameters:
        x (tf.Tensor, keras.layers.Layer): Input layer or tensor.
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
        multi_head_attention_dropout_rate (float): Multi-head attention dropout rate.
        return_attention_scores (bool): Whether to return attention scores or not.
        use_causal_mask: Whether to use casual mask in the multi-head attention.
        activation (str, keras.layers.Activation): The activation function of the output of the last hidden layer.
        sequential_activation (str): The activation function of the output of the sequential hidden layer.
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
        pooling (str, keras.layers.Layer): Pooling type.
        pool_size (int, Tuple[int]): Pooling size.
        sequential_axis (int): The axis that defines the sequence. For sequential models is normally 1. For
        self-attention (num_heads > 0) and image-like inputs, the sequential axis is the channel axis (3 for 2D
        images and 4 for 3D images).
        kernel_regularizer_l1 (float): Kernel regularization using l1 penalization (Lasso).
        kernel_regularizer_l2 (float): Kernel regularization using l2 penalization (Ridge).
        bias_regularizer_l1 (float): Bias regularization using l1 penalization (Lasso).
        bias_regularizer_l2 (float): Bias regularization using l2 penalization (Ridge).
        dropout_rate (float): Dropout rate.
        normalization_type (str): If set, a normalization layer (BN or LN) will be added right before the output activation function.

    Returns:
        x (keras.layers.Layer | tuple(keras.layers.Layer, keras.layers.Layer)): A keras layer.
    """

    if num_heads > 0 and units is None and key_dim == 0:
        warnings.warn(
            "in order to use a multi-head attention layer either units or key_dim needs to be set",
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
        convolutional_layer_name = "".join([name, "convolution"])
        x = convolutional_layer_constructor(
            x,
            name=convolutional_layer_name,
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
        pooling_layer_name = "".join([name, "pooling"])
        x = pooling_layer_constructor(
            x,
            name=pooling_layer_name,
            name_ext=name_ext,
            conv_transpose=conv_transpose,
            pooling=pooling,
            **pooling_kwargs,
        )

    # Multi-Head Attention
    attention_scores = None
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
            dropout=multi_head_attention_dropout_rate,
        )
        self_attention_layer_name = "".join([name, "self_attention"])
        x = self_attention_layer_constructor(
            x,
            name=self_attention_layer_name,
            name_ext=name_ext,
            sequential_axis=sequential_axis,
            return_attention_scores=return_attention_scores,
            use_causal_mask=use_causal_mask,
            **multi_head_attention_kwargs,
        )
        if return_attention_scores:
            x, attention_scores = x

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
            activation=sequential_activation,
        )
        sequential_layer_name = "".join([name, "sequential"])
        x = sequential_layer_constructor(
            x,
            name=sequential_layer_name,
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
        dense_layer_name = "".join([name, "dense"])
        x = dense_layer_constructor(
            x,
            name=dense_layer_name,
            name_ext=name_ext,
            **dense_kwargs,
        )

    # Normalization
    if normalization_type is None:
        pass
    elif normalization_type == "bn":
        bn_layer_name = "".join([name, "bn"])
        x = bn_layer_constructor(x, name=bn_layer_name, name_ext=name_ext)
    elif normalization_type == "ln":
        ln_layer_name = "".join([name, "ln"])
        x = ln_layer_constructor(x, name=ln_layer_name, name_ext=name_ext)
    else:
        raise ValueError(
            f"Unknown normalization type {normalization_type}. Only 'bn' and 'ln' are supported."
        )

    # Dropout
    if dropout_rate != 0:
        dropout_layer_name = "".join([name, "dropout"])
        x = dropout_layer_constructor(
            x,
            name=dropout_layer_name,
            name_ext=name_ext,
            dropout_rate=dropout_rate,
        )

    # Activation
    activation_kwargs = dict(
        alpha_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
    )
    activation_layer_name = "".join([name, "activation"])
    x = activation_layer_constructor(
        x,
        name=activation_layer_name,
        name_ext=name_ext,
        activation=activation,
        **activation_kwargs,
    )

    if return_attention_scores:
        return x, attention_scores
    else:
        return x


def dropout_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer],
    dropout_rate: float,
    name: str = "dropout",
    name_ext: Optional[str] = None,
) -> keras.layers.Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        output_reshape = input_shape
        flatten_layer_name = "_".join([name, "pre", "dropout", "flatten"])
        if name_ext is not None:
            flatten_layer_name = "_".join([flatten_layer_name, name_ext])
        x = keras.layers.Flatten(name=flatten_layer_name)(x)
    custom_dropout_layer_name = "_".join([name, "dropout"])
    if name_ext is not None:
        custom_dropout_layer_name = "_".join([custom_dropout_layer_name, name_ext])
    x = CustomDropout(name=custom_dropout_layer_name, rate=dropout_rate)(x)
    if output_reshape is not None:
        reshape_layer_name = "_".join([name, "reshape"])
        if name_ext is not None:
            reshape_layer_name = "_".join([reshape_layer_name, name_ext])
        x = keras.layers.Reshape(
            name=reshape_layer_name,
            target_shape=output_reshape[1:],
        )(x)
    return x


def convolutional_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer],
    name: str = "convolution",
    name_ext: Optional[str] = None,
    conv_transpose: bool = False,
    **kwargs,
) -> keras.layers.Layer:
    assert len(x.shape) <= 5, "x layer shape should have 5 or less dimensions"
    conv_dim = len(x.shape) - 2
    conv_type = "transpose" if conv_transpose else ""
    conv_name = "_".join(
        [name, "convolution" + str(conv_dim) + "d" + conv_type.capitalize()]
    )
    if name_ext is not None:
        conv_name = "_".join([conv_name, name_ext])
    if conv_dim == 1:
        conv_layer = keras.layers.Conv1D
    elif conv_dim == 2:
        conv_layer = keras.layers.Conv2D
    else:
        conv_layer = keras.layers.Conv3D
    x = conv_layer(
        name=conv_name,
        **kwargs,
    )(x)
    return x


def pooling_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer],
    name: str = "pooling",
    name_ext: Optional[str] = None,
    pooling: Union[str, keras.layers.Layer] = "max",
    **kwargs,
) -> keras.layers.Layer:
    if isinstance(pooling, str):
        pooling_name = pooling.capitalize() + "Pooling" + str(_get_pooling_dim(x)) + "D"
        pooling_layer = globals()[pooling_name]
    else:
        pooling_name = [k for k, v in locals().iteritems() if v == pooling][0]
        pooling_layer = pooling
    pooling_layer_name = "_".join([name, pooling_name.lower()])
    if name_ext is not None:
        pooling_layer_name = "_".join([pooling_layer_name, name_ext])
    x = pooling_layer(name=pooling_layer_name)(x)
    return x


def _get_pooling_dim(x: keras.layers.Layer) -> int:
    return len(x.shape) - 2


def self_attention_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer],
    name: str = "self_attention",
    name_ext: Optional[str] = None,
    sequential_axis: int = 1,
    return_attention_scores: bool = False,
    use_causal_mask: bool = False,
    **kwargs,
) -> keras.layers.Layer:
    sequential_permutation_name = "_".join(
        [name, "sequential_permutation_" + str(sequential_axis)]
    )
    x = sequential_permutation(
        x=x,
        name=sequential_permutation_name,
        name_ext=name_ext,
        sequential_axis=sequential_axis,
    )
    attention_layer_name = "_".join([name, "multi_head_attention"])
    if name_ext is not None:
        attention_layer_name = "_".join([attention_layer_name, name_ext])
    attention_layer = keras.layers.MultiHeadAttention(
        name=attention_layer_name,
        **kwargs,
    )
    if return_attention_scores:
        attention_layer.build(
            query_shape=tf.keras.backend.int_shape(x),
            value_shape=tf.keras.backend.int_shape(x),
        )
        x = attention_layer.call(
            query=x,
            value=x,
            key=x,
            use_causal_mask=use_causal_mask,
            return_attention_scores=return_attention_scores,
        )
    else:
        x = attention_layer(
            query=x,
            value=x,
            key=x,
            use_causal_mask=use_causal_mask,
            return_attention_scores=return_attention_scores,
        )
    return x


def sequential_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer],
    name: str = "sequential",
    name_ext: Optional[str] = None,
    bidirectional: bool = False,
    sequential_axis: int = 1,
    **kwargs,
) -> keras.layers.Layer:
    sequential_permutation_name = "_".join(
        [name, "sequential_permutation_" + str(sequential_axis)]
    )
    x = sequential_permutation(
        x=x,
        name=sequential_permutation_name,
        name_ext=name_ext,
        sequential_axis=sequential_axis,
    )
    gru_layer_name = "_".join([name, "gru"])
    if name_ext is not None:
        gru_layer_name = "_".join([gru_layer_name, name_ext])
    if bidirectional:
        x = keras.layers.Bidirectional(keras.layers.GRU(name=gru_layer_name, **kwargs))(
            x
        )
    else:
        x = keras.layers.GRU(name=gru_layer_name, **kwargs)(x)
    return x


def dense_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer],
    name: str = "dense",
    name_ext: Optional[str] = None,
    **kwargs,
) -> keras.layers.Layer:
    input_shape = x.shape
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        flatten_layer_name = "_".join([name, "pre", "dense", "flatten"])
        if name_ext is not None:
            flatten_layer_name = "_".join([flatten_layer_name, name_ext])
        x = keras.layers.Flatten(
            name=flatten_layer_name,
        )(x)
    dense_layer_name = "_".join([name, "dense"])
    if name_ext is not None:
        dense_layer_name = "_".join([dense_layer_name, name_ext])
    x = keras.layers.Dense(name=dense_layer_name, **kwargs)(x)
    return x


def bn_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer], name: str, name_ext: str, **kwargs
) -> keras.layers.Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        output_reshape = input_shape
        flatten_layer_name = "_".join([name, "pre", "bn", "flatten"])
        if name_ext is not None:
            flatten_layer_name = "_".join([flatten_layer_name, name_ext])
        x = keras.layers.Flatten(name=flatten_layer_name)(x)
    bn_layer_name = "_".join([name, "batch_normalization"])
    if name_ext is not None:
        bn_layer_name = "_".join([bn_layer_name, name_ext])
    x = keras.layers.BatchNormalization(
        name=bn_layer_name,
        **kwargs,
    )(x)
    if output_reshape is not None:
        reshape_layer_name = "_".join([name, "post", "bn", "reshape"])
        if name_ext is not None:
            reshape_layer_name = "_".join([reshape_layer_name, name_ext])
        x = keras.layers.Reshape(
            name=reshape_layer_name,
            target_shape=output_reshape[1:],
        )(x)
    return x


def ln_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer], name: str, name_ext: str, **kwargs
) -> keras.layers.Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        output_reshape = input_shape
        flatten_layer_name = "_".join([name, "pre", "ln", "flatten"])
        if name_ext is not None:
            flatten_layer_name = "_".join([flatten_layer_name, name_ext])
        x = keras.layers.Flatten(name=flatten_layer_name)(x)
    ln_layer_name = "_".join([name, "layer_normalization"])
    if name_ext is not None:
        ln_layer_name = "_".join([ln_layer_name, name_ext])
    x = keras.layers.LayerNormalization(
        name=ln_layer_name,
        **kwargs,
    )(x)
    if output_reshape is not None:
        reshape_layer_name = "_".join([name, "post", "ln", "reshape"])
        if name_ext is not None:
            reshape_layer_name = "_".join([reshape_layer_name, name_ext])
        x = keras.layers.Reshape(
            name=reshape_layer_name,
            target_shape=output_reshape[1:],
        )(x)
    return x


def activation_layer_constructor(
    x: Union[tf.Tensor, keras.layers.Layer],
    name: str = "activation",
    name_ext: Optional[str] = None,
    activation: str = "linear",
    **reg_kwargs,
) -> keras.layers.Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        output_reshape = input_shape
        flatten_layer_name = "_".join([name, "pre", "activation", "flatten"])
        if name_ext is not None:
            flatten_layer_name = "_".join([flatten_layer_name, name_ext])
        x = keras.layers.Flatten(name=flatten_layer_name)(x)
    activation_layer_name = "_".join(
        [name, activation if isinstance(activation, str) else "activation"],
    )
    if name_ext is not None:
        activation_layer_name = "_".join([activation_layer_name, name_ext])
    if activation == "leakyrelu":
        x = keras.layers.LeakyReLU(name=activation_layer_name)(x)
    elif activation == "prelu":
        x = keras.layers.PReLU(name=activation_layer_name, **reg_kwargs)(x)
    elif activation == "softmax":
        x = keras.layers.Softmax(name=activation_layer_name, dtype="float32")(x)
    else:
        x = keras.layers.Activation(name=activation_layer_name, activation=activation)(
            x
        )
    if output_reshape is not None:
        reshape_layer_name = "_".join([name, "post", "activation", "reshape"])
        if name_ext is not None:
            reshape_layer_name = "_".join([reshape_layer_name, name_ext])
        x = keras.layers.Reshape(
            name=reshape_layer_name,
            target_shape=output_reshape[1:],
        )(x)
    return x


def sequential_permutation(
    x: Union[tf.Tensor, keras.layers.Layer],
    name: str = "permutation",
    name_ext: Optional[str] = None,
    sequential_axis: int = 1,
) -> keras.layers.Layer:
    input_shape = x.shape
    sequential_axis_ = list(range(len(input_shape)))[sequential_axis]
    if sequential_axis_ != 1:
        permutation = tuple(
            [sequential_axis_]
            + [i for i in range(1, len(input_shape[1:])) if i != sequential_axis_],
        )
        permute_layer_name = "_".join([name, "permutation"])
        if name_ext is not None:
            permute_layer_name = "_".join([permute_layer_name, name_ext])
        x = keras.layers.Permute(name=permute_layer_name, dims=permutation)(x)
    else:
        permutation = list(range(1, len(input_shape)))
    if len(input_shape) > 2 and all([shape is not None for shape in input_shape[1:]]):
        permutation_dimensions = [input_shape[i] for i in permutation[1:]]
        reshape_layer_name = "_".join([name, "reshape"])
        if name_ext is not None:
            reshape_layer_name = "_".join([reshape_layer_name, name_ext])
        x = keras.layers.Reshape(
            name=reshape_layer_name,
            target_shape=(
                input_shape[permutation[0]],
                int(np.prod(permutation_dimensions)),
            ),
        )(x)
    return x


def identity(x: keras.layers.Layer) -> keras.layers.Layer:
    """Identity layer."""
    return x


class CustomDropout(keras.layers.Dropout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rate = tf.Variable(
            self.rate,
            trainable=False,
            name="_".join([self.name, "rate"]),
            dtype=self.dtype,
        )
