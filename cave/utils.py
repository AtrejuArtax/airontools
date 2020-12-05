from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU, PReLU, Input, BatchNormalization, Dense, Dropout, Activation, GRU, \
    Bidirectional, Concatenate, Reshape, Conv1D, Flatten, Softmax, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.python.ops import init_ops
import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def customized_net(specs, net_name='', compile_model=True, metrics=None):

    # Set precision
    if 'float16' in specs['precision']:
        if specs['precision'] == 'mixed_float16':
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        else:
            tf.keras.backend.set_floatx('float16')

    # Make the ensemble of models
    inputs = []
    outputs = []
    for device in specs['device']:

        # Define output dimension
        o_dim = sum([output_specs['dim'] for _, output_specs in specs['output_specs'].items()])

        # Device name
        device_name = device.replace('/', '').replace(':', '')

        # Parallel models per device
        for parallel_model in np.arange(0, specs['parallel_models']):

            # Initializations of blocks
            i_blocks, c_block, o_blocks, to_l = [], [], [], []

            # Name
            name = device_name + '_' + net_name + '_' + str(parallel_model)

            # Input Blocks
            for i_name, i_specs in specs['input_specs'].items():
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
                if i_specs['sequential'] and not specs['sequential_block']:
                    x_ = Conv1D(name=i_block_name + '_cnn1d',
                                filters=int(K.int_shape(x_)[2] / 2) + 1,
                                kernel_size=int(K.int_shape(x_)[1] / 2) + 1,
                                use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(
                                   l1=specs['kernel_regularizer_l1'],
                                   l2=specs['kernel_regularizer_l2']),
                                bias_regularizer=regularizers.l1_l2(
                                   l1=specs['bias_regularizer_l1'],
                                   l2=specs['bias_regularizer_l2']))(x_)
                    x_ = Flatten(name=i_block_name + '_flatten')(x_)
                    x_ = Dense(name=i_block_name + '_dense',
                               units=i_specs['dim'],
                               use_bias=True,
                               kernel_initializer=init_ops.random_normal_initializer(),
                               bias_initializer=init_ops.zeros_initializer(),
                               kernel_regularizer=regularizers.l1_l2(
                                   l1=specs['kernel_regularizer_l1'],
                                   l2=specs['kernel_regularizer_l2']),
                               bias_regularizer=regularizers.l1_l2(
                                   l1=specs['bias_regularizer_l1'],
                                   l2=specs['bias_regularizer_l2']))(x_)
                inputs += [x]

                # Define input block units
                i_units = get_layer_units(
                    input_dim=i_specs['dim'],
                    output_dim=i_specs['dim'] if not 'compression' in specs.keys() else\
                        int(i_specs['dim'] * (1 - specs['compression']) + 1),
                    n_layers=specs['i_n_layers'])[1:]
                i_units = rm_redundant(values=i_units, value=1)

                # Post Input Block
                sequential = specs['sequential_block'] and i_specs['sequential']
                length = None if not i_specs['sequential'] else i_specs['length']
                bidirectional = specs['bidirectional'] if sequential else False
                to_l += [len(i_units)]
                i_block = custom_block(units=i_units,
                                       name=i_block_name,
                                       specs=specs,
                                       input_shape=tuple([d for d in x.shape][1:]),
                                       sequential=sequential,
                                       length=length,
                                       bidirectional=bidirectional)
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
                n_layers=specs['c_n_layers'])[1:]

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
            for o_name, o_specs in specs['output_specs'].items():
                o_block_name = name + '_' + o_name
                o_block = custom_block(units=[o_dim],
                                       name=o_block_name,
                                       specs=specs,
                                       input_shape=tuple([d for d in c_block.shape][1:]),
                                       from_l=from_l,
                                       activation=specs['output_activation'])
                outputs += [o_block(c_block)]

    # Define model and compile
    model = models.Model(inputs=inputs, outputs=outputs)

    if compile_model:

        # Metrics
        metrics_ = []
        if metrics == 'accuracy':
            metrics_ += [tf.keras.metrics.Accuracy()]
        elif metrics == 'categorical_accuracy':
            metrics_ += [tf.keras.metrics.CategoricalAccuracy()]
        elif metrics == 'auc':
            metrics_ += [tf.keras.metrics.AUC()]

        # Compile
        model.compile(optimizer=Adam(learning_rate=specs['lr']),
                      loss=specs['loss'],
                      metrics=metrics_)

    return model


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
    model = models.Model(inputs=i_l, outputs=o_l)

    return model


def customized_layer(x, input_shape, units, activation, specs, name, l, dropout=True, return_sequences=False,
                     sequential=False, bidirectional=False):

    # Dropout
    if dropout:
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
    if activation == 'leakyrelu':
        x = LeakyReLU(
            name = name + '_' + activation + '_' + str(l),
            input_shape=input_shape,
            alpha=specs['alpha'])(x)
    elif activation == 'prelu':
        x = PReLU(
            name=name + '_' + activation + '_' + str(l),
            input_shape=input_shape,
            alpha_regularizer=regularizers.l1_l2(
                l1=specs['bias_regularizer_l1'],
                l2=specs['bias_regularizer_l2']))(x)
    elif activation == 'softmax':
        x = Softmax(
            name=name + '_' + activation + '_' + str(l),
            input_shape=input_shape,
            dtype='float32')(x)
    else:
        x = Activation(
            name=name + '_' + activation + '_' + str(l),
            input_shape=input_shape,
            activation=activation)(x)

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
