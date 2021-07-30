from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, Reshape, Conv1D, Flatten, LeakyReLU, PReLU, Input, BatchNormalization,\
    Dense, Dropout, Activation, GRU, Bidirectional, Softmax
from tensorflow.keras import models
from tensorflow.python.ops import init_ops
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


def custom_block(units, name, specs, input_shape, sequential=False, length=None, bidirectional=False, from_l=1,
                 activation=None):

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
            activation=specs['hidden_activation'] if activation is None else activation,
            specs=specs,
            name=name,
            l=l,
            sequential=sequential,
            return_sequences=True if l < to_l - 1 and sequential else False,
            bidirectional=bidirectional)
        pre_o_dim = o_dim

    # Model
    model = models.Model(inputs=i_l, outputs=o_l, name=name)

    return model


def customized_layer(x, input_shape, units, specs, name, l, activation=None, return_sequences=False,
                     sequential=False, bidirectional=False):

    # Dropout
    if specs['dropout_rate'] != 0:
        x = Dropout(
            name=name + '_dropout_' + str(l),
            rate=specs['dropout_rate'],
            input_shape=input_shape)(x)

    # Recurrent
    if sequential:
        if bidirectional:
            x = Bidirectional(GRU(
                name=name + '_gru_' + str(l),
                input_shape=input_shape,
                units=units,
                use_bias=True,
                kernel_initializer=init_ops.random_normal_initializer(),
                bias_initializer=init_ops.zeros_initializer(),
                kernel_regularizer=regularizers.l1_l2(
                    l1=specs['kernel_regularizer_l1'],
                    l2=specs['kernel_regularizer_l2']),
                bias_regularizer=regularizers.l1_l2(
                    l1=specs['bias_regularizer_l1'],
                    l2=specs['bias_regularizer_l2']),
                return_sequences=return_sequences,
                activation='linear'),
            input_shape=input_shape)(x)
        else:
            x = GRU(
                name=name + '_gru_' + str(l),
                input_shape=input_shape,
                units=units,
                use_bias=True,
                kernel_initializer=init_ops.random_normal_initializer(),
                bias_initializer=init_ops.zeros_initializer(),
                kernel_regularizer=regularizers.l1_l2(
                    l1=specs['kernel_regularizer_l1'],
                    l2=specs['kernel_regularizer_l2']),
                bias_regularizer=regularizers.l1_l2(
                    l1=specs['bias_regularizer_l1'],
                    l2=specs['bias_regularizer_l2']),
                return_sequences=return_sequences,
                activation='linear')(x)

    # Dense
    else:
        x = Dense(
            name=name + '_dense_' + str(l),
            input_shape=input_shape,
            units=units,
            use_bias=True,
            kernel_initializer=init_ops.random_normal_initializer(),
            bias_initializer=init_ops.zeros_initializer(),
            kernel_regularizer=regularizers.l1_l2(
                l1=specs['kernel_regularizer_l1'],
                l2=specs['kernel_regularizer_l2']),
            bias_regularizer=regularizers.l1_l2(
                l1=specs['bias_regularizer_l1'],
                l2=specs['bias_regularizer_l2']))(x)

    # Batch Normalization
    if specs['bn']:
        x = BatchNormalization(
            name=name + '_batch_normalization_' + str(l),
            input_shape=input_shape)(x)

    # Activation
    activation_ = activation if activation else 'prelu'
    if activation_ == 'leakyrelu':
        x = LeakyReLU(
            name = name + '_' + activation_ + '_' + str(l),
            input_shape=input_shape,
            alpha=specs['alpha'])(x)
    elif activation_ == 'prelu':
        x = PReLU(
            name=name + '_' + activation_ + '_' + str(l),
            input_shape=input_shape,
            alpha_regularizer=regularizers.l1_l2(
                l1=specs['bias_regularizer_l1'],
                l2=specs['bias_regularizer_l2']))(x)
    elif activation_ == 'softmax':
        x = Softmax(
            name=name + '_' + activation_ + '_' + str(l),
            input_shape=input_shape,
            dtype='float32')(x)
    else:
        if isinstance(activation_, str):
            x = Activation(
                name=name + '_' + activation_ + '_' + str(l),
                input_shape=input_shape,
                activation=activation_)(x)
        else:
            x = activation_(
                name=name + '_' + activation_.name + '_' + str(l),
                input_shape=input_shape)(x)

    return x


def to_time_series(tensor):
    return K.expand_dims(tensor, axis=2)


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


def get_layer_units(input_dim, output_dim, n_layers):
    n_units = input_dim if input_dim >= output_dim else output_dim
    return [int(units) for units in np.linspace(n_units, output_dim, n_layers + 1)]


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


def net_constructor(input_specs, output_specs, devices, net_name='', compile_model=True, metrics=None, lr=0.001, **specs):

    precision = specs['precision'] if 'precision' in specs else 'float32'
    parallel_models = specs['parallel_models'] if 'parallel_models' in specs else 1
    sequential_block = specs['sequential_block'] if 'sequential_block' in specs else False
    kernel_regularizer_l1 = specs['kernel_regularizer_l1'] if 'kernel_regularizer_l1' in specs else 0
    kernel_regularizer_l2 = specs['kernel_regularizer_l2'] if 'kernel_regularizer_l2' in specs else 0
    bias_regularizer_l1 = specs['bias_regularizer_l1'] if 'bias_regularizer_l1' in specs else 0
    bias_regularizer_l2 = specs['bias_regularizer_l2'] if 'bias_regularizer_l2' in specs else 0
    compression = specs['compression'] if 'compression' in specs else 0
    i_n_layers = specs['i_n_layers'] if 'i_n_layers' in specs else 1
    c_n_layers = specs['c_n_layers'] if 'c_n_layers' in specs else 1
    bidirectional = specs['bidirectional'] if 'bidirectional' in specs else False
    output_activation = specs['output_activation'] if 'output_activation' in specs else 'linear'
    optimizer = specs['optimizer'] if 'optimizer' in specs else Adam(learning_rate=lr)
    loss = specs['loss'] if 'loss' in specs else 'mse'

    # Set precision
    if 'float16' in precision:
        if precision == 'mixed_float16':
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        else:
            tf.keras.backend.set_floatx('float16')

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
            name = device_name + '_' + net_name + '_' + str(parallel_model)

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
                    x_ = Reshape((K.int_shape(x)[-1],), name=i_block_name + '_cat_reshape')(x)
                else:
                    x_ = x
                if i_specs['sequential'] and not sequential_block:
                    x_ = Conv1D(name=i_block_name + '_cnn1d',
                                filters=int(K.int_shape(x_)[2] / 2) + 1,
                                kernel_size=int(K.int_shape(x_)[1] / 2) + 1,
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
                               kernel_initializer=init_ops.random_normal_initializer(),
                               bias_initializer=init_ops.zeros_initializer(),
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
                                       specs=specs,
                                       input_shape=tuple([d for d in x.shape][1:]),
                                       sequential=sequential,
                                       length=length,
                                       bidirectional=bidirectional_)
                i_blocks += [i_block(x_)]

            # Concat input blocks
            if len(i_blocks) > 1:
                i_blocks = Concatenate(name='i_block_conc', axis=-1)(i_blocks)
            else:
                i_blocks = i_blocks[0]

            # Define core block units
            c_units = get_layer_units(
                input_dim=K.int_shape(i_blocks)[-1],
                output_dim=o_dim + 1,
                n_layers=c_n_layers)[1:]

            # Core block
            from_l = max(to_l)
            to_l = from_l + len(c_units)
            c_block_name = name + '_c_block'
            c_block = custom_block(units=c_units,
                                   name=c_block_name,
                                   specs=specs,
                                   input_shape=tuple([d for d in i_blocks.shape][1:]),
                                   from_l=from_l)
            c_block = c_block(i_blocks)

            # Output Blocks
            from_l = to_l + len(c_units)
            for o_name, o_specs in output_specs.items():
                o_block_name = name + '_' + o_name
                o_block = custom_block(units=[o_dim],
                                       name=o_block_name,
                                       specs=specs,
                                       input_shape=tuple([d for d in c_block.shape][1:]),
                                       from_l=from_l,
                                       activation=output_activation)
                outputs += [o_block(c_block)]

    # Define model and compile
    model = models.Model(inputs=inputs, outputs=outputs, name=net_name)

    if compile_model:

        # Metrics
        metrics_ = []
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
