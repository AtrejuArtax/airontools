import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as k_bcknd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from airontools.constructors.utils.utils_tf import set_precision
from airontools.constructors.utils.utils_tf import get_regularizer, get_layer_units, rm_redundant
from airontools.constructors.layers import layer_constructor
from airontools.constructors.blocks import block_constructor


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
