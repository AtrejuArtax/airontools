from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras import regularizers
import tensorflow.keras.backend as k_bcknd
import numpy as np


def get_latent_model(model, layer_names):
    layer_names = layer_names if isinstance(layer_names, list) else [layer_names]
    return Model(inputs=model.inputs,
                 outputs=[layer.output for layer in model.layers
                          if any([layer_name in layer.name for layer_name in layer_names])])


def set_precision(precision):
    if 'float16' in precision:
        if precision == 'mixed_float16':
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        else:
            tf.keras.backend.set_floatx('float16')


def to_time_series(tensor):
    return k_bcknd.expand_dims(tensor, axis=2)


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


def get_layer_units(input_dim, output_dim, n_layers, min_hidden_units=2):
    units = [max(int(units), min_hidden_units) for units in np.linspace(input_dim, output_dim, n_layers + 1)]
    units[0], units[-1] = input_dim, output_dim
    return units


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


def get_regularizer(l1=None, l2=None):
    if l1 and l2:
        regularizer = regularizers.l1_l2(l1=l1, l2=l2)
    elif l1:
        regularizer = regularizers.l1(l1)
    elif l2:
        regularizer = regularizers.l2(l2)
    else:
        regularizer = None
    return regularizer