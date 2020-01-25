import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, layers
from tensorflow.keras import models
from tensorflow.python.ops import init_ops
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'claudi'


def customized_net(specs, net_name='', compile_model=True, metrics=list([])):

    inputs = []
    outputs = []
    for device in specs['device']:

        # Device name
        device_name = device.translate(None, '/:')

        # Assign device
        with tf.device(device):

            # Parallel models per device
            for parallel_model in np.arange(0, specs['parallel_models']):

                # Name
                name = device_name + '_' + net_name + '_' + str(parallel_model)

                # Input layer
                x = layers.Input(
                    shape=(specs['units'][0],),
                    name=name + '_input')

                # Dense layers
                output = models.Sequential()
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
    x.add(keras.layers.Dropout(
        name=name + '_dropout' + '_' + str(i),
        rate=specs['dropout_rate'],
        input_shape=(input_dim,)))

    # Dense
    x.add(keras.layers.Dense(
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
    x.add(keras.layers.BatchNormalization(
        name=name + '_batch_normalization_' + str(i),
        input_shape=(input_dim,)))

    # Activation
    if activation == 'leakyrelu':
        x.add(keras.layers.LeakyReLU(
            name = name + '_' + activation + '_' + str(i),
            input_shape=(input_dim,),
            alpha=specs['alpha']))
    else:
        x.add(keras.layers.Activation(
            name = name + '_' + activation + '_' + str(i),
            input_shape=(input_dim,),
            activation=activation))

    return x

def plot_loss(losses):
    plt.figure(figsize=(10, 8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()


def train_children(specs, gim_model, generator, discriminator, children, losses):

    # Generate from GIM
    counts = 0
    gim_samples = None
    children_gms_outputs = None
    children_gss_outputs = None
    while counts < specs['minimum_batch_for_training']:
        gim_sample = gim_model.sample(specs['batch_size'])

        if counts == 0:
            gim_samples = gim_sample

        # Output from children model
        predictions = children.predict(gim_sample)
        children_gms_output = predictions[0]
        children_gss_output = predictions[1]

        if children_gss_output < specs['p_tmp_threshold']:
            gim_samples = np.stack((gim_samples, gim_sample))
            children_gms_outputs = np.stack((children_gms_outputs, children_gms_output))
            children_gss_outputs = np.stack((children_gss_outputs, children_gss_output))

    # Train discriminator model
    make_trainable(discriminator, True)
    d_loss = discriminator.train_on_batch(
        x=np.vstack((children_gms_outputs, gim_samples)),
        y=np.vstack((np.zeros(children_gms_outputs.shape), np.ones(children_gms_outputs.shape))))
    losses["d"].append(d_loss)

    # Train children model
    actual_p = None # TO DO
    make_trainable(discriminator, False)
    g_loss = children.train_on_batch(
        x=gim_samples,
        y=[np.zeros(children_gms_outputs.shape), actual_p])
    losses["g"].append(g_loss)


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def children_nets(specss):

    # GSM model
    gsm = customized_net(
        specs=specss['gsm_model'],
        net_name='gsm_model',
        compile_model=False)

    # GMS model
    gms = customized_net(
        specs=specss['gms_model'],
        net_name='gms_model',
        compile_model=False)

    # GSS model
    gss = customized_net(
        specs=specss['gss_model'],
        net_name='gss_model',
        compile_model=False)

    # Generator model
    generator = models.Model(
        inputs=gsm.inputs,
        outputs=[gms(gsm.outputs), gss(gsm.outputs)])
    generator.compile(
        optimizer=specss['optimizer'],
        loss='mse')

    # Discriminative model
    discriminator = customized_net(
        specs=specss['disc_model'],
        net_name='disc_model',
        compile_model=False)
    discriminator.compile(
        optimizer=specss['optimizer'],
        loss='binary_crossentropy')

    # Children model
    make_trainable(discriminator, False)
    children_input = layers.Input(shape=gsm.input_shape)
    children = models.Model(children_input, discriminator(generator(children_input)[0]))
    children.compile(
        optimizer=specss['optimizer'],
        loss='binary_crossentropy')

    return generator, discriminator, children