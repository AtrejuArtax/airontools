from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape, Conv1D, Flatten
from tensorflow.keras import models
from tensorflow.python.ops import init_ops
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from airontools.net_constructors.utils_tf import get_layer_units, rm_redundant, custom_block


def net_constructor(specs, net_name='', compile_model=True, metrics=None):

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
        model.compile(optimizer=specs['optimizer'],
                      loss=specs['loss'],
                      metrics=metrics_)

    return model