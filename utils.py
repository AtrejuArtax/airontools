import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU, PReLU, Input, BatchNormalization, Dense, Dropout, Activation, GRU, \
    Lambda, Bidirectional
from tensorflow.keras import models
from tensorflow.python.ops import init_ops
import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import classification_report


def customized_net(specs, net_name='', compile_model=True, metrics=None):

    # Define units per hidden layer
    if specs['sequential']:
        n_units = 1
    else:
        n_units = specs['n_input']
    specs['units'] = [specs['n_input']] + [int(units)
                                           for units in np.linspace(n_units, specs['n_output'], specs['n_layers'])]

    inputs = []
    outputs = []
    for device in specs['device']:

        # Device name
        device_name = device.replace('/', '').replace(':', '')

        # Assign device
        with tf.device(device):

            # Parallel models per device
            for parallel_model in np.arange(0, specs['parallel_models']):

                # Name
                name = device_name + '_' + net_name + '_' + str(parallel_model)

                # Input layer
                x = Input(
                    shape=(specs['units'][0],),
                    name=name + '_input')

                # Hidden layers
                output = models.Sequential(name=name)
                if specs['sequential']: # To time series
                    output.add(Lambda(
                        to_time_series,
                        name=name + '_totimeseries'))
                i = 0
                for i in range(1, len(specs['units']) - 1):
                    output = customized_layer(
                        x=output,
                        input_dim=specs['units'][i-1],
                        units=specs['units'][i],
                        activation=specs['hidden_activation'],
                        specs=specs,
                        name=name,
                        i=i,
                        return_sequences=True,
                        bidirectional=specs['bidirectional'])

                # Output Layer
                i += 1
                output = customized_layer(
                    x=output,
                    input_dim=specs['units'][-2],
                    units=specs['units'][-1],
                    activation=specs['output_activation'],
                    specs=specs,
                    name=name + '_output',
                    i=i,
                    dropout=False)

                # Inputs and outputs
                inputs += [x]
                outputs += [output(x)]

    # Define model and compile
    model = models.Model(
        inputs=inputs,
        outputs=outputs)

    if compile_model:

        # Compile
        model.compile(
            optimizer=specs['optimizer'],
            loss=specs['loss'],
            metrics=metrics)

    return model


def customized_layer(x, input_dim, units, activation, specs, name, i, dropout=True, return_sequences=False,
                     bidirectional=False):

    # Input shape
    if specs['sequential']:
        input_shape = (specs['length'], input_dim,)
    else:
        input_shape = (input_dim,)

    # Dropout
    if dropout:
        x.add(Dropout(
            name=name + '_dropout' + '_' + str(i),
            rate=specs['dropout_rate'],
            input_shape=input_shape))

    # Recurrent
    if specs['sequential']:
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
    return cat_encoder.inverse_transform(model.predict(x))

