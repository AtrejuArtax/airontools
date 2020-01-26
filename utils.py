import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU, PReLU, Input, BatchNormalization,Dense, Dropout, Activation
from tensorflow.keras import models
from tensorflow.python.ops import init_ops
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'claudi'


def customized_net(specs, net_name='', compile_model=True, metrics=None):

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

                # Dense layers
                output = models.Sequential(name=name)
                i = 0
                for i in range(1, len(specs['units']) - 1):
                    output = customized_dense(
                        x=output,
                        input_dim=specs['units'][i-1],
                        units=specs['units'][i],
                        activation=specs['hidden_activation'],
                        specs=specs,
                        name=name,
                        i=i)

                # Output Layer
                i += 1
                output = customized_dense(
                    x=output,
                    input_dim=specs['units'][-2],
                    units=specs['units'][-1],
                    activation=specs['output_activation'],
                    specs=specs,
                    name=name + '_output',
                    i=i)

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


def customized_dense(x, input_dim, units, activation, specs, name, i):

    # Dropout
    x.add(Dropout(
        name=name + '_dropout' + '_' + str(i),
        rate=specs['dropout_rate'],
        input_shape=(input_dim,)))

    # Dense
    x.add(Dense(
        name=name + '_dense_' + str(i),
        input_shape=(input_dim,),
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
            input_shape=(input_dim,)))

    # Activation
    if activation == 'leakyrelu':
        x.add(LeakyReLU(
            name = name + '_' + activation + '_' + str(i),
            input_shape=(input_dim,),
            alpha=specs['alpha']))
    if activation == 'prelu':
        x.add(PReLU(
            name = name + '_' + activation + '_' + str(i),
            input_shape=(input_dim,)))
    else:
        x.add(Activation(
            name = name + '_' + activation + '_' + str(i),
            input_shape=(input_dim,),
            activation=activation))

    return x

def plot_loss(losses):
    plt.figure(figsize=(10, 8))
    plt.plot(losses, label='loss')
    plt.legend()
    plt.show()


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
