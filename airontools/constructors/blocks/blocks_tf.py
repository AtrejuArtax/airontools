from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from airontools.constructors.layers import layer_constructor


def block_constructor(units,
                      input_shape,
                      name=None,
                      sequential=False,
                      length=None,
                      bidirectional=False,
                      from_l=1,
                      hidden_activation=None,
                      output_activation=None,
                      advanced_reg=False,
                      **reg_kwargs):
    """ It builds a custom block. reg_kwargs contain everything regarding regularization.

        Parameters:
            units (list): Number of units per hidden layer.
            input_shape (tuple): Input shape.
            name (str): Name of the block.
            sequential (bool): Whether to consider a sequential model or not.
            length (int): Length of the sequence (only active if sequential).
            bidirectional (bool): Whether to consider bidirectional case or not (only active if sequential).
            from_l (int): The number indicator of the first hidden layer of the block, useful to make sure that layer
            names are not repeated.
            hidden_activation (str, Layer): Hidden activation function.
            output_activation (str, Layer): The activation function of the output of the block.
            advanced_reg (bool): Whether to automatically set advanced regularization. Useful to quickly make use of all
            the regularization properties.
            dropout_rate (float): Probability of each intput being disconnected.
            kernel_regularizer_l1 (float): Kernel regularization using l1 penalization (Lasso).
            kernel_regularizer_l2 (float): Kernel regularization using l2 penalization (Ridge).
            bias_regularizer_l1 (float): Bias regularization using l1 penalization (Lasso).
            bias_regularizer_l2 (float): Bias regularization using l2 penalization (Ridge).
            bn (bool): If set, a batch normalization layer will be added right before the output activation function.

        Returns:
            model (Model): A keras model.
    """

    # Initializations
    name = name if name else 'block'
    hidden_activation = hidden_activation if hidden_activation else 'prelu'
    output_activation = output_activation if output_activation else 'linear'

    # Hidden layers
    i_l, o_l = Input(shape=input_shape, name=''.join([name, 'input'])), None
    to_l = from_l + len(units)
    pre_o_dim = None
    for l, o_dim in zip(range(from_l, to_l), units):
        if l > from_l:
            input_shape = (length, pre_o_dim,) if sequential else (pre_o_dim,)
        else:
            o_l = i_l
        o_l = layer_constructor(
            x=o_l,
            input_shape=input_shape,
            units=o_dim,
            activation=hidden_activation if l == to_l - 1 is None else output_activation,
            name=name,
            name_ext=str(l),
            sequential=sequential,
            return_sequences=True if l < to_l - 1 and sequential else False,
            bidirectional=bidirectional,
            advanced_reg=advanced_reg,
            **reg_kwargs)
        pre_o_dim = o_dim

    # Model
    model = Model(inputs=i_l, outputs=o_l, name=name)

    return model
