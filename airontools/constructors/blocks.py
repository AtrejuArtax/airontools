from typing import List, Tuple, Union

import tensorflow as tf

from airontools.constructors.layers import layer_constructor


def block_constructor(
    units: List[int],
    input_shape: Tuple[int],
    name: str = "block",
    sequential: bool = False,
    bidirectional: bool = False,
    from_l: int = 1,
    hidden_activation: str = "prelu",
    output_activation: Union[str, tf.keras.layers.Layer] = "linear",
    kernel_regularizer_l1: float = 0.001,
    kernel_regularizer_l2: float = 0.001,
    bias_regularizer_l1: float = 0.001,
    bias_regularizer_l2: float = 0.001,
    dropout_rate: float = 0.0,
    bn: bool = False,
) -> tf.keras.models.Model:
    """It builds a custom block. For now only compatible with dense and sequential layers.

    Parameters:
        units (list): Number of units per hidden layer.
        input_shape (tuple): Input shape.
        name (str): Name of the block.
        sequential (bool): Whether to consider a sequential model or not.
        bidirectional (bool): Whether to consider bidirectional case or not (only active if sequential).
        from_l (int): The number indicator of the first hidden layer of the block, useful to make sure that layer
        names are not repeated.
        hidden_activation (str, Layer): Hidden activation function.
        output_activation (str, Layer): The activation function of the output of the block.
        dropout_rate (float): Probability of each intput being disconnected.
        kernel_regularizer_l1 (float): Kernel regularization using l1 penalization (Lasso).
        kernel_regularizer_l2 (float): Kernel regularization using l2 penalization (Ridge).
        bias_regularizer_l1 (float): Bias regularization using l1 penalization (Lasso).
        bias_regularizer_l2 (float): Bias regularization using l2 penalization (Ridge).
        bn (bool): If set, a batch normalization layer will be added right before the output activation function.

    Returns:
        model (Model): A keras model.
    """

    # Hidden layers
    i_l, o_l = (
        tf.keras.layers.Input(shape=input_shape, name="_".join([name, "input"])),
        None,
    )
    to_l = from_l + len(units)
    for l, o_dim in zip(range(from_l, to_l), units):
        if l <= from_l:
            o_l = i_l
        o_l = layer_constructor(
            x=o_l,
            units=o_dim,
            activation=hidden_activation
            if l == to_l - 1 is None
            else output_activation,
            name=name,
            name_ext=str(l),
            sequential=sequential,
            return_sequences=True if l < to_l - 1 and sequential else False,
            bidirectional=bidirectional,
            kernel_regularizer_l1=kernel_regularizer_l1,
            kernel_regularizer_l2=kernel_regularizer_l2,
            bias_regularizer_l1=bias_regularizer_l1,
            bias_regularizer_l2=bias_regularizer_l2,
            dropout_rate=dropout_rate,
            bn=bn,
        )

    # Model
    model = tf.keras.models.Model(inputs=i_l, outputs=o_l, name=name)

    return model
