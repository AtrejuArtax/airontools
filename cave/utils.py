import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU, PReLU, Input, BatchNormalization, Dense, Dropout, Activation, GRU, \
    Bidirectional, Concatenate, Reshape
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

        # Define number of units per layer
        specs['units'] = get_layer_units(
            input_dim=sum([input_specs['dim'] for _, input_specs in specs['input_specs'].items()]),
            output_dim=sum([input_specs['dim'] for _, input_specs in specs['output_specs'].items()]),
            n_layers=specs['n_layers'],
            compression=None if not 'compression' in specs.keys() else specs['compression'])

        # Device name
        device_name = device.replace('/', '').replace(':', '')

        # Assign device
        with tf.device(device):

            # Parallel models per device
            for parallel_model in np.arange(0, specs['parallel_models']):

                # Initializations of blocks
                input_blocks, output_blocks = [], []

                # Name
                name = device_name + '_' + net_name + '_' + str(parallel_model)

                # Input Blocks
                i = None
                for input_name, input_specs in specs['input_specs'].items():
                    i_block_name = name + '_' + input_name
                    if input_specs['sequential']:
                        input_shape = (input_specs['length'], input_specs['dim'],)
                    elif not input_specs['sequential'] and input_specs['type'] == 'cat':
                        input_shape = (1, input_specs['dim'],)
                    else:
                        input_shape = (input_specs['dim'],)
                    x = Input(shape=input_shape,
                              name=i_block_name + '_input')
                    if not input_specs['sequential'] and input_specs['type'] == 'cat':
                        x_ = Reshape((K.int_shape(x)[-1],), name=i_block_name + '_Reshape')(x)
                    else:
                        x_ = x
                    inputs += [x]

                    # Hidden layers
                    input_block = models.Sequential(name=i_block_name)
                    from_i = 1
                    for i in range(from_i, 2):
                        input_block = customized_layer(
                            x=input_block,
                            input_dim=input_specs['dim'] if i == from_i else specs['units'][i-1],
                            units=specs['units'][i],
                            activation=specs['hidden_activation'],
                            specs=specs,
                            name=i_block_name,
                            i=i,
                            sequential=input_specs['sequential'],
                            length=None if not input_specs['sequential'] else input_specs['length'],
                            return_sequences=True,
                            bidirectional=specs['bidirectional'] if input_specs['sequential'] else False)
                    input_blocks += [input_block(x_)]

                # Concat input blocks
                if len(input_blocks) > 1:
                    input_blocks = Concatenate(name=i_block_name + '_conc', axis=-1)(input_blocks)
                else:
                    input_blocks = input_blocks[0]

                # Core block
                core_block = models.Sequential(name=i_block_name)
                from_i = 2
                for i in range(from_i, len(specs['units']) - 1):
                    core_block = customized_layer(
                        x=core_block,
                        input_dim=K.int_shape(input_blocks)[-1] if i == from_i else K.int_shape(core_block)[-1],
                        units=specs['units'][i],
                        activation=specs['hidden_activation'],
                        specs=specs,
                        name=i_block_name,
                        i=i,
                        sequential=input_specs['sequential'],
                        length=None if not input_specs['sequential'] else input_specs['length'],
                        return_sequences=True,
                        bidirectional=specs['bidirectional'] if input_specs['sequential'] else False)
                core_block = core_block(input_blocks)

                # Output Blocks
                from_i = len(specs['units']) - 1
                for output_name, output_specs in specs['output_specs'].items():
                    o_block_name = name + '_' + output_name
                    output_block = models.Sequential(name=o_block_name)
                    output_block = customized_layer(
                        x=output_block,
                        input_dim=K.int_shape(core_block)[-1],
                        units=specs['units'][-1],
                        activation=specs['output_activation'],
                        specs=specs,
                        name=o_block_name + '_output',
                        i=from_i,
                        dropout=False)
                    outputs += [output_block(core_block)]

    # Define model and compile
    model = models.Model(inputs=inputs, outputs=outputs)

    if compile_model:

        # Compile
        model.compile(optimizer=Adam(learning_rate=specs['lr']),
                      loss=specs['loss'],
                      metrics=metrics)

    return model


def customized_layer(x, input_dim, units, activation, specs, name, i, dropout=True, return_sequences=False,
                     sequential=False, length=None, bidirectional=False):

    # Input shape
    if sequential:
        input_shape = (length, input_dim,)
    else:
        input_shape = (input_dim,)

    # Dropout
    if dropout:
        x.add(Dropout(
            name=name + '_dropout' + '_' + str(i),
            rate=specs['dropout_rate'],
            input_shape=input_shape))

    # Recurrent
    if sequential:
        if bidirectional:
            x.add(Bidirectional(GRU(
                name=name + '_gru_' + str(i),
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
                name=name + '_gru_' + str(i),
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
            name=name + '_dense_' + str(i),
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
            name=name + '_batch_normalization_' + str(i),
            input_shape=input_shape))

    # Activation
    if activation == 'leakyrelu':
        x.add(LeakyReLU(
            name = name + '_' + activation + '_' + str(i),
            input_shape=input_shape,
            alpha=specs['alpha']))
    if activation == 'prelu':
        x.add(PReLU(
            name = name + '_' + activation + '_' + str(i),
            input_shape=input_shape))
    else:
        x.add(Activation(
            name = name + '_' + activation + '_' + str(i),
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


def get_layer_units(input_dim, output_dim, n_layers, compression=None):
    n_units = input_dim if compression is None else int(input_dim * (1 - compression) + 1)
    n_units = n_units if n_units >= output_dim else output_dim
    return [input_dim] + [int(units) for units in np.linspace(n_units, output_dim, n_layers + 2)][1:]
