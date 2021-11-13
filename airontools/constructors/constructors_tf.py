from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as k_bcknd
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from airontools.constructors.utils_tf import set_precision
from airontools.constructors.utils_tf import get_regularizer, get_layer_units, rm_redundant


def layer_constructor(x, name=None, name_ext=None, units=None, num_heads=None, key_dim=None, value_dim=None,
                      activation=None, use_bias=True, sequential=False, bidirectional=False, return_sequences=False,
                      filters=None, kernel_size=None, padding='valid', pooling=None, pool_size=None,
                      conv_transpose=False, strides=(1, 1), advanced_reg=False, **reg_kwargs):
    """ It builds a custom layer. reg_kwargs contain everything regarding regularization. For now only 2D convolutions
    are supported for input of rank 4. ToDo: add transformers.

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
            sequential (bool): Whether to consider a sequential custom layer or not.
            bidirectional (bool): Whether to consider bidirectional case or not (only active if sequential).
            names are not repeated.
            return_sequences (bool): Whether to return sequences or not (only active if sequential).
            filters (int): Number of filters to be used. If a value is given, a convolutional layer will be
            automatically added.
            kernel_size (int): Kernel size for the convolutional layer.
            conv_transpose (bool): Whether to use a transpose conv layer or not (only active if filters and
            kernel_size are set).
            strides (tuple, int): Strides for the conv layer (only active if filters and
            kernel_size are set).
            padding (str): Padding to be applied (only active if filters and
            kernel_size are set).
            pooling (str, layer): Pooling type.
            pool_size (int, tuple): Pooling size.
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
    input_shape = tuple(list(x.shape[1:]),)
    conv_condition = all([conv_param is not None for conv_param in [filters, kernel_size]])
    if conv_condition and len(input_shape) == 1:
        warnings.warn('if filters and kernel are set then the shape of x should be rank 4')
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
        if not len(input_shape) == 1:
            x = Flatten(name='_'.join([name, 'predropout_flatten', name_ext]))(x)
        x = Dropout(
            name='_'.join([name, 'dropout', name_ext]),
            rate=dropout_rate,
            input_shape=input_shape)(x)

    # Convolution
    if conv_condition:
        if len(x.shape[1:]) == 1:
            x = Reshape(name='_'.join([name, 'preconv_reshape', name_ext]), target_shape=input_shape)(x)
        conv_kwargs = dict(use_bias=use_bias,
                           filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           kernel_regularizer=get_regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
                           bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2))
        conv_dim = len(x.shape) - 2
        conv_type = 'transpose' if conv_transpose else ''
        conv_name = 'Conv' + str(conv_dim) + 'D' + conv_type.capitalize()
        x = globals()[conv_name](
            name='_'.join([name, conv_name.lower(), name_ext]), **conv_kwargs)(x)

    # Pooling
    if pooling is not None:
        pooling_dim = len(x.shape) - 2
        pooling_kwargs = dict(pool_size=pool_size if pool_size is not None else tuple([2] * pooling_dim),
                              strides=strides,
                              padding=padding)
        if isinstance(pooling, str):
            pooling_name = pooling.capitalize() + 'Pooling' + str(pooling_dim) + 'D'
            pooling_layer = globals()[pooling_name]
        else:
            pooling_name = [k for k, v in locals().iteritems() if v == pooling][0]
            pooling_layer = pooling
        x = pooling_layer(name='_'.join([name, pooling_name.lower(), name_ext]))(x)

    # Recurrent
    if sequential:
        seq_kwargs = dict(input_shape=input_shape,
                          units=units,
                          use_bias=use_bias,
                          kernel_regularizer=get_regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
                          bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
                          return_sequences=return_sequences,
                          activation='linear')
        if bidirectional:
            x = Bidirectional(GRU(
                name='_'.join([name, 'gru', name_ext]),
                **seq_kwargs),
                input_shape=input_shape)(x)
        else:
            x = GRU(
                name='_'.join([name, 'gru', name_ext]),
                **seq_kwargs)(x)

    # Dense
    elif units:
        if len(x.shape[1:]) != 1:
            x = Flatten(name='_'.join([name, 'predense_flatten', name_ext]))(x)
        x = Dense(
            name='_'.join([name, 'dense', name_ext]),
            input_shape=input_shape,
            units=units,
            use_bias=use_bias,
            kernel_regularizer=get_regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
            bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2))(x)

    # Multi-Head Attention
    if num_heads is not None:
        x = MultiHeadAttention(
            name='_'.join([name, 'multi_head_attention', name_ext]),
            key_dim=key_dim if key_dim is not None else units,
            value_dim=value_dim if value_dim is not None else key_dim,
            use_bias=use_bias,
            kernel_regularizer=get_regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
            bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2))(x)

    # Pre output reshape
    post_output_shape = None
    if len(x.shape[1:]) != 1:
        post_output_shape = x.shape[1:]
        x = Flatten(name='_'.join([name, 'pre_output_flatten', name_ext]))(x)

    # Batch Normalization
    if bn:
        x = BatchNormalization(
            name='_'.join([name, 'batch_normalization', name_ext]),
            input_shape=input_shape)(x)

    # Activation
    activation_name = '_'.join([name, activation if isinstance(activation, str) else 'activation', name_ext])
    if activation == 'leakyrelu':
        x = LeakyReLU(
            name=activation_name,
            input_shape=input_shape)(x)
    elif activation == 'prelu':
        x = PReLU(
            name=activation_name,
            input_shape=input_shape,
            alpha_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2))(x)
    elif activation == 'softmax':
        x = Softmax(
            name=activation_name,
            input_shape=input_shape,
            dtype='float32')(x)
    else:
        if isinstance(activation, str):
            x = Activation(
                name=activation_name,
                input_shape=input_shape,
                activation=activation)(x)
        else:
            x = activation(
                name=activation_name,
                input_shape=input_shape)(x)

    # Post output reshape
    if post_output_shape:
        x = Reshape(name='_'.join([name, 'post_output_reshape', name_ext]), target_shape=post_output_shape)(x)

    return x


def block_constructor(units, input_shape, name=None, sequential=False, length=None, bidirectional=False, from_l=1,
                      hidden_activation=None, output_activation=None, advanced_reg=False, **reg_kwargs):
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


def model_constructor(input_specs, output_specs, name=None, optimizer=None, lr=0.001, loss='mse', i_n_layers=1,
                      c_n_layers=1, hidden_activation=None, output_activation=None, i_compression=None,
                      sequential=False, bidirectional=False, parallel_models=1, precision='float32', devices=None,
                      compile_model=True, metrics=None, advanced_reg=False, **reg_kwargs):
    """ It builds a custom model. reg_kwargs contain everything regarding regularization.

        Parameters:
            input_specs (dict): Input specifications.
            output_specs (dict): Output specifications.
            name (str): Name of the custom layer.
            optimizer (str): Name of the custom layer.
            lr (float): Learning rate.
            loss (str): Loss.
            i_n_layers (int): Number of layers per input block.
            c_n_layers (int): Number of layers in the core block.
            hidden_activation (str, Layer): Hidden activation function.
            output_activation (str, Layer): The activation function of the output of the block.
            i_compression (float): Input block compression.
            sequential (bool): Whether to consider a sequential custom model or not.
            bidirectional (bool): Whether to consider bidirectional case or not (only active if sequential).
            parallel_models (int): Number of parallel models.
            precision (str): Precision to be considered for the model: 'float32', 'float16' or 'mixed_float16'.
            devices (list): Physical devises for training/inference.
            compile_model (bool): Whether to compile the model or not.
            metrics (list): Metrics. Useful for when the loss is not enough for hyper-parameter optimisation.
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
    name = name if name is not None else 'NN'
    optimizer = optimizer if optimizer else Adam(learning_rate=lr)
    hidden_activation = hidden_activation if hidden_activation else 'prelu'
    output_activation = output_activation if output_activation else 'linear'

    # Regularization parameters
    kernel_regularizer_l1 = reg_kwargs['kernel_regularizer_l1'] if 'kernel_regularizer_l1' in reg_kwargs.keys() \
        else 0.001 if advanced_reg else None
    kernel_regularizer_l2 = reg_kwargs['kernel_regularizer_l2'] if 'kernel_regularizer_l2' in reg_kwargs.keys() \
        else 0.001 if advanced_reg else None
    bias_regularizer_l1 = reg_kwargs['bias_regularizer_l1'] if 'bias_regularizer_l1' in reg_kwargs.keys() \
        else 0.001 if advanced_reg else None
    bias_regularizer_l2 = reg_kwargs['bias_regularizer_l2'] if 'bias_regularizer_l2' in reg_kwargs.keys() \
        else 0.001 if advanced_reg else None

    # Set precision
    set_precision(precision)

    # Make the ensemble of models
    inputs = []
    outputs = []
    for device in devices:

        # Define output dimension
        o_dim = sum([output_specs_['dim'] for _, output_specs_ in output_specs.items()])

        # Device name
        device_name = device.replace('/', '').replace(':', '')

        # Parallel models per device
        for parallel_model in np.arange(0, parallel_models):

            # Initializations of blocks
            i_blocks, c_block, o_blocks, to_l = [], [], [], []

            # Name
            name_ = '_'.join([device_name, name, str(parallel_model)])

            # Input Blocks
            for i_name, i_specs in input_specs.items():
                i_block_name = '_'.join([name_, i_name, 'input_block'])
                if i_specs['sequential']:
                    i_shape = (i_specs['length'], i_specs['dim'],)
                elif not i_specs['sequential'] and i_specs['type'] == 'cat':
                    i_shape = (1, i_specs['dim'],)
                else:
                    i_shape = (i_specs['dim'],)
                x = Input(shape=i_shape,
                          name='_'.join([i_block_name, 'input']))
                if not i_specs['sequential'] and i_specs['type'] == 'cat':
                    x_ = Reshape((k_bcknd.int_shape(x)[-1],), name='_'.join([i_block_name, 'cat_reshape']))(x)
                else:
                    x_ = x
                if i_specs['sequential'] and not sequential:
                    x_ = Conv1D(name='_'.join([i_block_name, 'cnn1d']),
                                filters=int(k_bcknd.int_shape(x_)[2] / 2) + 1,
                                kernel_size=int(k_bcknd.int_shape(x_)[1] / 2) + 1,
                                use_bias=True,
                                kernel_regularizer=get_regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
                                bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2))(x_)
                    x_ = Flatten(name='_'.join([i_block_name, 'flatten']))(x_)
                    x_ = layer_constructor(x_,
                                           name=i_block_name,
                                           units=i_specs['dim'],
                                           activation=hidden_activation,
                                           **reg_kwargs)
                inputs += [x]

                # Define input block units
                i_units = get_layer_units(
                    input_dim=i_specs['dim'],
                    output_dim=i_specs['dim'] if i_compression is None else int(i_specs['dim'] * (1 - i_compression) + 1),
                    n_layers=i_n_layers)[1:]
                i_units = rm_redundant(values=i_units, value=1)

                # Post Input Block
                sequential_ = sequential and i_specs['sequential']
                length = None if not i_specs['sequential'] else i_specs['length']
                bidirectional_ = bidirectional if sequential else False
                to_l += [len(i_units)]
                i_block = block_constructor(units=i_units,
                                            name=i_block_name,
                                            input_shape=tuple([d for d in x.shape][1:]),
                                            sequential=sequential_,
                                            length=length,
                                            bidirectional=bidirectional_,
                                            hidden_activation=hidden_activation,
                                            output_activation=hidden_activation,
                                            **reg_kwargs)
                i_blocks += [i_block(x_)]

            # Concat input blocks
            if len(i_blocks) > 1:
                i_blocks = Concatenate(name='input_block_conc', axis=-1)(i_blocks)
            else:
                i_blocks = i_blocks[0]

            # Define core block units
            c_units = get_layer_units(
                input_dim=k_bcknd.int_shape(i_blocks)[-1],
                output_dim=o_dim + 1,
                n_layers=c_n_layers)[1:]

            # Core block
            from_l = max(to_l)
            to_l = from_l + len(c_units)
            c_block = block_constructor(units=c_units,
                                        name='_'.join([name_, 'core_block']),
                                        input_shape=tuple([d for d in i_blocks.shape][1:]),
                                        from_l=from_l,
                                        hidden_activation=hidden_activation,
                                        output_activation=hidden_activation,
                                        **reg_kwargs)
            c_block = c_block(i_blocks)

            # Output Blocks
            from_l = to_l + len(c_units)
            for o_name, o_specs in output_specs.items():
                o_block = block_constructor(units=[o_dim],
                                            name='_'.join([name_, o_name]),
                                            input_shape=tuple([d for d in c_block.shape][1:]),
                                            from_l=from_l,
                                            hidden_activation=hidden_activation,
                                            output_activation=output_activation,
                                            **reg_kwargs)
                outputs += [o_block(c_block)]

    # Define model and compile
    model = Model(inputs=inputs, outputs=outputs, name=name)

    if compile_model:

        # Metrics
        metrics_ = []
        if metrics:
            if 'accuracy' in metrics:
                metrics_ += [tf.keras.metrics.Accuracy()]
            elif 'categorical_accuracy' in metrics:
                metrics_ += [tf.keras.metrics.CategoricalAccuracy()]
            elif 'auc' in metrics:
                metrics_ += [tf.keras.metrics.AUC()]

        # Compile
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics_)

    return model
