from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as k_bcknd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from airontools.model_constructors.utils_tf import set_precision


def custom_block(units, input_shape, name=None, sequential=False, length=None, bidirectional=False, from_l=1,
                 output_activation='linear', hidden_activation='prelu', advanced_params=False, **reg_kwargs):
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
            output_activation (str): The activation function of the output of the block.
            hidden_activation (str): Hidden activation function.
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

    # Hidden layers
    i_l, o_l = Input(shape=input_shape, name=name + '_input'), None
    to_l = from_l + len(units)
    pre_o_dim = None
    for l, o_dim in zip(range(from_l, to_l), units):
        if l > from_l:
            input_shape = (length, pre_o_dim,) if sequential else (pre_o_dim,)
        else:
            o_l = i_l
        o_l = customized_layer(
            x=o_l,
            input_shape=input_shape,
            units=o_dim,
            activation=hidden_activation if l == to_l - 1 is None else output_activation,
            name=name,
            name_ext=str(l),
            sequential=sequential,
            return_sequences=True if l < to_l - 1 and sequential else False,
            bidirectional=bidirectional,
            advanced_params=advanced_params,
            **reg_kwargs)
        pre_o_dim = o_dim

    # Model
    model = Model(inputs=i_l, outputs=o_l, name=name)

    return model


def customized_layer(x, name='', name_ext='', units=None, activation='prelu', sequential=False, bidirectional=False,
                     return_sequences=False, filters=None, kernel_size=None, conv_transpose=False, strides=(1,1),
                     advanced_reg=False, **reg_kwargs):
    """ It builds a custom layer. reg_kwargs contain everything regarding regularization. For now only 2D convolutions
    are supported for input of rank 4. ToDo: add transformers.

        Parameters:
            x (Layer): Input layer.
            name (str): Name of the custom layer.
            name_ext (str): Extension name for the custom layer that will be at the end of of it.
            units (int): Number of units for the dense layer. If a value is given, a dense layer will be added
            automatically if not sequential, else a sequential model. Useful to force an output dimensionality of the
            custom layer when using convolutional layers.
            activation (str): The activation function of the output of the last hidden layer.
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

    input_shape = tuple(list(x.shape[1:]),)

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
            x = Flatten(name=name + 'predropout_flatten' + name_ext)(x)
        x = Dropout(
            name=name + 'dropout' + name_ext,
            rate=dropout_rate,
            input_shape=input_shape)(x)

    # Convolution
    if all([conv_param is not None for conv_param in [filters, kernel_size]]):
        if len(x.shape[1:]) == 1:
            x = Reshape(name=name + 'preconv_reshape' + name_ext, target_shape=input_shape)(x)
        conv_kwargs = dict(input_shape=input_shape,
                           filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           use_bias=True,
                           kernel_regularizer=get_regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
                           bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2))
        if conv_transpose:
            x = Conv2DTranspose(name=name + 'conv2d' + name_ext,
                                **conv_kwargs)(x)
        else:
            x = Conv2D(name=name + 'conv2d' + name_ext,
                       **conv_kwargs)(x)

    # Recurrent
    if sequential:
        seq_kwargs = dict(input_shape=input_shape,
                          units=units,
                          use_bias=True,
                          kernel_regularizer=get_regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
                          bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2),
                          return_sequences=return_sequences,
                          activation='linear')
        if bidirectional:
            x = Bidirectional(GRU(
                name=name + 'gru' + name_ext,
                **seq_kwargs),
                input_shape=input_shape)(x)
        else:
            x = GRU(
                name=name + 'gru' + name_ext,
                **seq_kwargs)(x)

    # Dense
    elif units:
        if len(x.shape[1:]) != 1:
            x = Flatten(name=name + 'predense_flatten' + name_ext)(x)
        x = Dense(
            name=name + 'dense' + name_ext,
            input_shape=input_shape,
            units=units,
            use_bias=True,
            kernel_regularizer=get_regularizer(kernel_regularizer_l1, kernel_regularizer_l2),
            bias_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2))(x)

    # Batch Normalization
    if bn:
        if len(x.shape[1:]) != 1:
            x = Flatten(name=name + 'prebn_flatten' + name_ext)(x)
        x = BatchNormalization(
            name=name + 'batch_normalization' + name_ext,
            input_shape=input_shape)(x)

    # Activation
    activation_ = activation if activation else PReLU
    if activation_ == 'leakyrelu':
        x = LeakyReLU(
            name=name + activation_ + name_ext,
            input_shape=input_shape)(x)
    elif activation_ == 'prelu':
        x = PReLU(
            name=name + activation_ + name_ext,
            input_shape=input_shape,
            alpha_regularizer=get_regularizer(bias_regularizer_l1, bias_regularizer_l2))(x)
    elif activation_ == 'softmax':
        x = Softmax(
            name=name + activation_ + name_ext,
            input_shape=input_shape,
            dtype='float32')(x)
    else:
        if isinstance(activation_, str):
            x = Activation(
                name=name + activation_ + name_ext,
                input_shape=input_shape,
                activation=activation_)(x)
        else:
            x = activation_(
                name=name + 'activation' + name_ext,
                input_shape=input_shape)(x)

    return x


def to_time_series(tensor):
    return k_bckndexpand_dims(tensor, axis=2)


def evaluate_clf(cat_encoder, model, x, y):
    pred = inference(cat_encoder, model, x)
    print("\nReport:")
    print(classification_report(y, pred, digits=4))


def inference(cat_encoder, model, x):
    inf = model.predict(x)
    if isinstance(inf, list):
        inf = [sub_inf.reshape(sub_inf.shape + tuple([1])) for sub_inf in inf]
        inf = np.concatenate(tuple(inf), axis=-1)
        inf = np.mean(inf, axis=-1)
    return cat_encoder.inverse_transform(inf)


def get_layer_units(input_dim, output_dim, n_layers, min_hidden_units=2):
    units = [max(int(units), min_hidden_units) for units in np.linspace(input_dim, output_dim, n_layers + 1)]
    units[0], units[-1] = input_dim, output_dim
    return units


def rm_redundant(values, value):
    taken = False
    values_ = []
    for n in values:
        if n != value:
            values_ += [n]
        elif not taken:
            values_ += [n]
            taken = True
    return values_


def model_constructor(input_specs, output_specs, devices, model_name='', compile_model=True, metrics=None, lr=0.001, **kwargs):

    precision = kwargs['precision'] if 'precision' in kwargs else 'float32'
    parallel_models = kwargs['parallel_models'] if 'parallel_models' in kwargs else 1
    sequential_block = kwargs['sequential_block'] if 'sequential_block' in kwargs else False
    kernel_regularizer_l1 = kwargs['kernel_regularizer_l1'] if 'kernel_regularizer_l1' in kwargs else 0
    kernel_regularizer_l2 = kwargs['kernel_regularizer_l2'] if 'kernel_regularizer_l2' in kwargs else 0
    bias_regularizer_l1 = kwargs['bias_regularizer_l1'] if 'bias_regularizer_l1' in kwargs else 0
    bias_regularizer_l2 = kwargs['bias_regularizer_l2'] if 'bias_regularizer_l2' in kwargs else 0
    compression = kwargs['compression'] if 'compression' in kwargs else 0
    i_n_layers = kwargs['i_n_layers'] if 'i_n_layers' in kwargs else 1
    c_n_layers = kwargs['c_n_layers'] if 'c_n_layers' in kwargs else 1
    bidirectional = kwargs['bidirectional'] if 'bidirectional' in kwargs else False
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs else 'linear'
    optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else Adam(learning_rate=lr)
    loss = kwargs['loss'] if 'loss' in kwargs else 'mse'

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
            name = device_name + '_' + model_name + '_' + str(parallel_model)

            # Input Blocks
            for i_name, i_specs in input_specs.items():
                i_block_name = name + '_' + i_name + '_i_block'
                if i_specs['sequential']:
                    i_shape = (i_specs['length'], i_specs['dim'],)
                elif not i_specs['sequential'] and i_specs['type'] == 'cat':
                    i_shape = (1, i_specs['dim'],)
                else:
                    i_shape = (i_specs['dim'],)
                x = Input(shape=i_shape,
                          name=i_block_name + '_input')
                if not i_specs['sequential'] and i_specs['type'] == 'cat':
                    x_ = Reshape((k_bckndint_shape(x)[-1],), name=i_block_name + '_cat_reshape')(x)
                else:
                    x_ = x
                if i_specs['sequential'] and not sequential_block:
                    x_ = Conv1D(name=i_block_name + '_cnn1d',
                                filters=int(k_bckndint_shape(x_)[2] / 2) + 1,
                                kernel_size=int(k_bckndint_shape(x_)[1] / 2) + 1,
                                use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(
                                   l1=kernel_regularizer_l1,
                                   l2=kernel_regularizer_l2),
                                bias_regularizer=regularizers.l1_l2(
                                   l1=bias_regularizer_l1,
                                   l2=bias_regularizer_l2))(x_)
                    x_ = Flatten(name=i_block_name + '_flatten')(x_)
                    x_ = Dense(name=i_block_name + '_dense',
                               units=i_specs['dim'],
                               use_bias=True,
                               kernel_regularizer=regularizers.l1_l2(
                                   l1=kernel_regularizer_l1,
                                   l2=kernel_regularizer_l2),
                               bias_regularizer=regularizers.l1_l2(
                                   l1=bias_regularizer_l1,
                                   l2=bias_regularizer_l2))(x_)
                inputs += [x]

                # Define input block units
                i_units = get_layer_units(
                    input_dim=i_specs['dim'],
                    output_dim=i_specs['dim'] if compression == 0 else int(i_specs['dim'] * (1 - compression) + 1),
                    n_layers=i_n_layers)[1:]
                i_units = rm_redundant(values=i_units, value=1)

                # Post Input Block
                sequential = sequential_block and i_specs['sequential']
                length = None if not i_specs['sequential'] else i_specs['length']
                bidirectional_ = bidirectional if sequential else False
                to_l += [len(i_units)]
                i_block = custom_block(units=i_units,
                                       name=i_block_name,
                                       input_shape=tuple([d for d in x.shape][1:]),
                                       sequential=sequential,
                                       length=length,
                                       bidirectional=bidirectional_,
                                       **kwargs)
                i_blocks += [i_block(x_)]

            # Concat input blocks
            if len(i_blocks) > 1:
                i_blocks = Concatenate(name='i_block_conc', axis=-1)(i_blocks)
            else:
                i_blocks = i_blocks[0]

            # Define core block units
            c_units = get_layer_units(
                input_dim=k_bckndint_shape(i_blocks)[-1],
                output_dim=o_dim + 1,
                n_layers=c_n_layers)[1:]

            # Core block
            from_l = max(to_l)
            to_l = from_l + len(c_units)
            c_block_name = name + '_c_block'
            c_block = custom_block(units=c_units,
                                   name=c_block_name,
                                   input_shape=tuple([d for d in i_blocks.shape][1:]),
                                   from_l=from_l,
                                   **kwargs)
            c_block = c_block(i_blocks)

            # Output Blocks
            from_l = to_l + len(c_units)
            for o_name, o_specs in output_specs.items():
                o_block_name = name + '_' + o_name
                o_block = custom_block(units=[o_dim],
                                       name=o_block_name,
                                       input_shape=tuple([d for d in c_block.shape][1:]),
                                       from_l=from_l,
                                       activation=output_activation,
                                       **kwargs)
                outputs += [o_block(c_block)]

    # Define model and compile
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

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


def get_regularizer(l1=None, l2=None):
    if l1 and l2:
        regularizer = regularizers.l1_l2(l1=l1, l2=l2)
    elif l1:
        regularizer = regularizers.l1(l1)
    elif l2:
        regularizer = regularizers.l2(l2)
    else:
        regularizer = None
    return regularizer
