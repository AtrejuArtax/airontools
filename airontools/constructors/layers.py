import warnings
from typing import Union

import numpy as np
from tensorflow.keras.layers import *

from airontools.constructors.utils import regularizer


def layer_constructor(x,
                      name=None,
                      name_ext=None,
                      units=None,
                      num_heads=None,
                      key_dim=None,
                      value_dim=None,
                      activation=None,
                      use_bias=True,
                      sequential=False,
                      bidirectional=False,
                      return_sequences=False,
                      filters=None,
                      kernel_size=None,
                      padding='valid',
                      pooling=None,
                      pool_size=None,
                      conv_transpose=False,
                      strides=(1, 1),
                      sequential_axis=1,
                      advanced_reg=False,
                      **reg_kwargs):
    """ It builds a custom layer. reg_kwargs contain everything regarding regularization. For now only 2D convolutions
    are supported for input of rank 4.

        Parameters:
            x (Layer): Input layer.
            name (str): Name of the custom layer.
            name_ext (str): Extension name for the custom layer that will be at the end of of it.
            units (int): Number of units for the dense layer. If a value is given, a dense layer will be added
            automatically if not sequential, else a sequential model. Useful to force an output dimensionality of the
            custom layer when using convolutional layers.
            num_heads (int): Number of heads for the multi-head attention layer.
            key_dim (int): Key dimensionality for the multi-head attention layer, if None then the number of units is
            used instead.
            value_dim (int): Value dimensionality for the multi-head attention layer, if None then key_dim is used
            instead.
            activation (str, Layer): The activation function of the output of the last hidden layer.
            use_bias (bool): Whether to sue bias or not.
            sequential (bool): Whether to consider a sequential custom layer or not. Sequential and self-attention
            (num_heads > 0) are not compatible.
            bidirectional (bool): Whether to consider bidirectional case or not (only active if sequential).
            names are not repeated.
            return_sequences (bool): Whether to return sequences or not (only active if sequential).
            filters (int): Number of filters to be used. If a value is given, a convolutional layer will be
            automatically added.
            kernel_size (int, tuple): Kernel size for the convolutional layer.
            conv_transpose (bool): Whether to use a transpose conv layer or not (only active if filters and
            kernel_size are set).
            strides (tuple, int): Strides for the conv layer (only active if filters and
            kernel_size are set).
            padding (str): Padding to be applied (only active if filters and
            kernel_size are set).
            pooling (str, layer): Pooling type.
            pool_size (int, tuple): Pooling size.
            sequential_axis (int): The axis that defines the sequence. For sequential models is normally 1. For
            self-attention (num_heads > 0) and image-like inputs, the sequential axis is the channel axis (3 for 2D
            images and 4 for 3D images).
            advanced_reg (bool): Whether to automatically set advanced regularization. Useful to quickly make use of all
            the regularization properties.
            dropout_rate (float): Probability of each intput being disconnected.
            kernel_regularizer_l1 (float): Kernel regularization using l1 penalization (Lasso).
            kernel_regularizer_l2 (float): Kernel regularization using l2 penalization (Ridge).
            bias_regularizer_l1 (float): Bias regularization using l1 penalization (Lasso).
            bias_regularizer_l2 (float): Bias regularization using l2 penalization (Ridge).
            bn (bool): If set, a batch normalization layer will be added right before the output activation function.

        Returns:
            x (Layer): A keras layer.
    """

    if num_heads is not None and units is None and key_dim is None:
        warnings.warn('in order to use a multi-head attention layer either units or key_dim needs to be set')

    # Initializations
    conv_condition = all([conv_param is not None for conv_param in [filters, kernel_size]])
    name = name if name else 'layer'
    name_ext = name_ext if name_ext else ''
    activation = activation if activation else 'prelu'

    # Regularization parameters
    dropout_rate = reg_kwargs['dropout_rate'] if 'dropout_rate' in reg_kwargs.keys() \
        else 0.1 if advanced_reg else 0
    kernel_regularizer_l1 = reg_kwargs['kernel_regularizer_l1'] if 'kernel_regularizer_l1' in reg_kwargs.keys() \
        else 0.001 if advanced_reg else None
    kernel_regularizer_l2 = reg_kwargs['kernel_regularizer_l2'] if 'kernel_regularizer_l2' in reg_kwargs.keys() \
        else 0.001 if advanced_reg else None
    bias_regularizer_l1 = reg_kwargs['bias_regularizer_l1'] if 'bias_regularizer_l1' in reg_kwargs.keys() \
        else 0.001 if advanced_reg else None
    bias_regularizer_l2 = reg_kwargs['bias_regularizer_l2'] if 'bias_regularizer_l2' in reg_kwargs.keys() \
        else 0.001 if advanced_reg else None
    bn = reg_kwargs['bn'] if 'bn' in reg_kwargs.keys() else True if advanced_reg else False

    # Dropout
    if dropout_rate != 0:
        x = dropout_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            dropout_rate=dropout_rate)

    # Convolution
    if conv_condition:
        conv_kwargs = dict(
            use_bias=use_bias,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_regularizer=regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
            bias_regularizer=regularizer(bias_regularizer_l1, bias_regularizer_l2))
        x = convolutional_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            conv_transpose=conv_transpose,
            **conv_kwargs)

    # Pooling
    if pooling is not None:
        pooling_kwargs = dict(
            pool_size=pool_size if pool_size is not None else tuple([2] * _get_pooling_dim(x)),
            strides=strides,
            padding=padding)
        x = pooling_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            conv_transpose=conv_transpose,
            pooling=pooling,
            **pooling_kwargs)

    # Multi-Head Attention
    if num_heads is not None:
        key_dim_ = key_dim if key_dim is not None else units
        value_dim_ = value_dim if value_dim is not None else key_dim_
        multi_head_attention_kwargs = dict(
            num_heads=num_heads,
            key_dim=key_dim_,
            value_dim=value_dim_,
            use_bias=use_bias,
            kernel_regularizer=regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
            bias_regularizer=regularizer(bias_regularizer_l1, bias_regularizer_l2))
        x = self_attention_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            sequential_axis=sequential_axis,
            **multi_head_attention_kwargs)

    # Sequential
    elif sequential:
        seq_kwargs = dict(
            units=units,
            use_bias=use_bias,
            kernel_regularizer=regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
            bias_regularizer=regularizer(bias_regularizer_l1, bias_regularizer_l2),
            return_sequences=return_sequences,
            activation='linear')
        x = sequential_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            bidirectional=bidirectional,
            sequential_axis=sequential_axis,
            **seq_kwargs)

    # Dense
    if units:
        dense_kwargs = dict(
            units=units,
            use_bias=use_bias,
            kernel_regularizer=regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
            bias_regularizer=regularizer(bias_regularizer_l1, bias_regularizer_l2))
        x = dense_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            **dense_kwargs)

    # Batch Normalization
    if bn:
        bn_kwargs = dict(
            beta_regularizer=regularizer(bias_regularizer_l1, bias_regularizer_l2),
            gamma_regularizer=regularizer(bias_regularizer_l1, bias_regularizer_l2))
        x = bn_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            **bn_kwargs)

    # Activation
    activation_kwargs = dict(
            alpha_regularizer=regularizer(bias_regularizer_l1, bias_regularizer_l2))
    x = activation_layer_constructor(
            x,
            name=name,
            name_ext=name_ext,
            activation=activation,
            **activation_kwargs)

    return x


def dropout_layer_constructor(x: Layer, name: str, name_ext: str, dropout_rate: float) -> Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2:
        output_reshape = input_shape
        x = Flatten(name='_'.join([name, 'pre', 'dropout', 'flatten', name_ext]))(x)
    x = Dropout(
        name='_'.join([name, 'dropout', name_ext]),
        rate=dropout_rate)(x)
    if output_reshape is not None:
        x = Reshape(name='_'.join([name, 'post', 'dropout', 'reshape', name_ext]),
                    target_shape=output_reshape[1:])(x)
    return x


def convolutional_layer_constructor(x: Layer, name: str, name_ext: str, conv_transpose: bool, **kwargs) -> Layer:
    conv_dim = len(x.shape) - 2
    conv_type = 'transpose' if conv_transpose else ''
    conv_name = 'Conv' + str(conv_dim) + 'D' + conv_type.capitalize()
    x = globals()[conv_name](
        name='_'.join([name, conv_name.lower(), name_ext]), **kwargs)(x)
    return x


def pooling_layer_constructor(x: Layer, name: str, name_ext: str, pooling: Union[str, Layer], **kwargs) -> Layer:
    if isinstance(pooling, str):
        pooling_name = pooling.capitalize() + 'Pooling' + str(_get_pooling_dim(x)) + 'D'
        pooling_layer = globals()[pooling_name]
    else:
        pooling_name = [k for k, v in locals().iteritems() if v == pooling][0]
        pooling_layer = pooling
    x = pooling_layer(name='_'.join([name, pooling_name.lower(), name_ext]))(x)
    return x


def _get_pooling_dim(x: Layer) -> int:
    return len(x.shape) - 2


def self_attention_layer_constructor(x: Layer, name: str, name_ext: str, sequential_axis: int, **kwargs) -> Layer:
    x = sequential_permutation(
        x=x,
        name=name,
        name_ext=name_ext,
        sequential_axis=sequential_axis
    )
    x = MultiHeadAttention(
        name='_'.join([name, 'multi_head_attention', name_ext]),
        **kwargs)(x, x)
    return x


def sequential_layer_constructor(x: Layer, name: str, name_ext: str, bidirectional: bool, sequential_axis: int,
                                 **kwargs) -> Layer:
    x = sequential_permutation(
        x=x,
        name=name,
        name_ext=name_ext,
        sequential_axis=sequential_axis
    )
    if bidirectional:
        x = Bidirectional(GRU(
            name='_'.join([name, 'gru', name_ext]),
            **kwargs))(x)
    else:
        x = GRU(
            name='_'.join([name, 'gru', name_ext]),
            **kwargs)(x)
    return x


def dense_layer_constructor(x: Layer, name: str, name_ext: str, **kwargs) -> Layer:
    if not len(x.shape[1:]) == 1:
        x = Flatten(name='_'.join([name, 'pre', 'dense', 'flatten', name_ext]))(x)
    x = Dense(
        name='_'.join([name, 'dense', name_ext]),
        **kwargs)(x)
    return x


def bn_layer_constructor(x: Layer, name: str, name_ext: str, **kwargs) -> Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2:
        output_reshape = input_shape
        x = Flatten(name='_'.join([name, 'pre', 'bn', 'flatten', name_ext]))(x)
    x = BatchNormalization(
        name='_'.join([name, 'batch_normalization', name_ext]),
        **kwargs)(x)
    if output_reshape is not None:
        x = Reshape(name='_'.join([name, 'post', 'bn', 'reshape', name_ext]),
                    target_shape=output_reshape[1:])(x)
    return x


def activation_layer_constructor(x: Layer, name: str, name_ext: str, activation: str, **reg_kwargs) -> Layer:
    input_shape = x.shape
    output_reshape = None
    if len(input_shape) > 2:
        output_reshape = input_shape
        x = Flatten(name='_'.join([name, 'pre', 'activation', 'flatten', name_ext]))(x)
    activation_name = '_'.join([name, activation if isinstance(activation, str) else 'activation', name_ext])
    if activation == 'leakyrelu':
        x = LeakyReLU(name=activation_name)(x)
    elif activation == 'prelu':
        x = PReLU(name=activation_name, **reg_kwargs)(x)
    elif activation == 'softmax':
        x = Softmax(
            name=activation_name,
            dtype='float32')(x)
    else:
        x = Activation(
            name=activation_name,
            activation=activation)(x)
    if output_reshape is not None:
        x = Reshape(name='_'.join([name, 'post', 'activation', 'reshape', name_ext]),
                    target_shape=output_reshape[1:])(x)
    return x


def sequential_permutation(x: Layer, name: str, name_ext: str, sequential_axis: int) -> Layer:
    input_shape = x.shape
    sequential_axis_ = list(range(len(input_shape)))[sequential_axis]
    if sequential_axis_ != 1:
        permutation = tuple([sequential_axis_] +
                            [i for i in range(1, len(input_shape[1:])) if i != sequential_axis_])
        x = Permute(
            name='_'.join([name, 'permutation', name_ext]),
            dims=permutation
        )(x)
    else:
        permutation = list(range(1, len(input_shape)))
    if len(input_shape) > 2:
        x = Reshape(
            name='_'.join([name, 'reshape', name_ext]),
            target_shape=(input_shape[permutation[0]], np.prod([input_shape[i] for i in permutation[1:]]),)
        )(x)
    return x


def identity(x) -> Layer:
    return x
