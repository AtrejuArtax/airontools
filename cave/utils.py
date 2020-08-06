import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU, PReLU, Input, BatchNormalization, Dense, Dropout, Activation, GRU, \
    Bidirectional, Concatenate, Reshape, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.python.ops import init_ops
import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import classification_report
tf.keras.backend.set_floatx('float16')


def customized_net(specs, net_name='', compile_model=True, metrics=None):

    # Make the ensemble of models
    inputs = []
    outputs = []
    for device in specs['device']:

        # Define output dimension
        o_dim = sum([input_specs['dim'] for _, input_specs in specs['output_specs'].items()])

        # Device name
        device_name = device.replace('/', '').replace(':', '')

        # Assign device
        with tf.device(device):

            # Parallel models per device
            for parallel_model in np.arange(0, specs['parallel_models']):

                # Initializations of blocks
                i_blocks, c_block, o_blocks = [], [], []

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

                    # Hidden layers
                    i_block = models.Sequential(name=i_block_name)
                    from_l = 1
                    to_l = from_l + len(i_units)
                    pre_n = None
                    sequential = specs['sequential_block'] and i_specs['sequential']
                    for l, n in zip(range(from_l, to_l), i_units):
                        i_block = customized_layer(
                            x=i_block,
                            input_dim=i_specs['dim'] if l == from_l else pre_n,
                            units=n,
                            activation=specs['hidden_activation'],
                            specs=specs,
                            name=i_block_name,
                            l=l,
                            sequential=sequential,
                            length=None if not i_specs['sequential'] else i_specs['length'],
                            return_sequences=True if l < to_l - 1 else False,
                            bidirectional=specs['bidirectional'] if sequential else False)
                        pre_n = n
                    i_blocks += [i_block(x_)]

                # Concat input blocks
                if len(i_blocks) > 1:
                    i_blocks = Concatenate(name=i_block_name + '_conc', axis=-1)(i_blocks)
                else:
                    i_blocks = i_blocks[0]

                # Define core block units
                c_units = get_layer_units(
                    input_dim=K.int_shape(i_blocks)[-1],
                    output_dim=o_dim + 1,
                    n_layers=specs['c_n_layers'])[1:]

                # Core block
                c_block_name = name + '_c_block'
                c_block = models.Sequential(name=c_block_name)
                from_l = to_l
                to_l = from_l + len(c_units)
                pre_n = None
                for l, n in zip(range(from_l, to_l), c_units):
                    c_block = customized_layer(
                        x=c_block,
                        input_dim=K.int_shape(i_blocks)[-1] if l == from_l else pre_n,
                        units=n,
                        activation=specs['hidden_activation'],
                        specs=specs,
                        name=c_block_name,
                        l=l)
                    pre_n = n
                c_block = c_block(i_blocks)

                # Output Blocks
                from_l = to_l
                for o_name, o_specs in specs['output_specs'].items():
                    o_block_name = name + '_' + o_name
                    o_block = models.Sequential(name=o_block_name)
                    o_block = customized_layer(
                        x=o_block,
                        input_dim=K.int_shape(c_block)[-1],
                        units=o_dim,
                        activation=specs['output_activation'],
                        specs=specs,
                        name=o_block_name + '_output',
                        l=from_l,
                        dropout=False)
                    outputs += [o_block(c_block)]

    # Define model and compile
    model = models.Model(inputs=inputs, outputs=outputs)

    if compile_model:

        # Compile
        model.compile(optimizer=Adam(learning_rate=specs['lr']),
                      loss=specs['loss'],
                      metrics=metrics)

    return model


def customized_layer(x, input_dim, units, activation, specs, name, l, dropout=True, return_sequences=False,
                     sequential=False, length=None, bidirectional=False):

    # Input shape
    if sequential:
        input_shape = (length, input_dim,)
    else:
        input_shape = (input_dim,)

    # Dropout
    if dropout:
        x.add(Dropout(
            name=name + '_dropout' + '_' + str(l),
            rate=specs['dropout_rate'],
            input_shape=input_shape))

    # Recurrent
    if sequential:
        if bidirectional:
            x.add(Bidirectional(GRU(
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
            input_shape=input_shape))
        else:
            x.add(GRU(
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
                activation='linear'))

    # Dense
    else:
        x.add(Dense(
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
                l2=specs['bias_regularizer_l2'])))

    # Batch Normalization
    if specs['bn']:
        x.add(BatchNormalization(
            name=name + '_batch_normalization_' + str(l),
            input_shape=input_shape))

    # Activation
    if activation == 'leakyrelu':
        x.add(LeakyReLU(
            name = name + '_' + activation + '_' + str(l),
            input_shape=input_shape,
            alpha=specs['alpha']))
    elif activation == 'prelu':
        x.add(PReLU(
            name = name + '_' + activation + '_' + str(l),
            input_shape=input_shape,
            alpha_regularizer=regularizers.l1_l2(
                l1=specs['bias_regularizer_l1'],
                l2=specs['bias_regularizer_l2'])))
    else:
        x.add(Activation(
            name = name + '_' + activation + '_' + str(l),
            input_shape=input_shape,
            activation=activation))

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
