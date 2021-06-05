from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU, PReLU, Input, BatchNormalization, Dense, Dropout, Activation, GRU, \
    Bidirectional, Softmax
from tensorflow.keras import models
from tensorflow.python.ops import init_ops
import tensorflow.keras.backend as K
import numpy as np
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
