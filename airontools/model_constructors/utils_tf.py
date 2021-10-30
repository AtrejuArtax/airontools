from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def get_latent_model(model, layer_names):
    layer_names = layer_names if isinstance(layer_names, list) else [layer_names]
    return Model(inputs=model.inputs,
                 outputs=[layer.outputs for layer in model.layers
                          if any([layer_name in layer.name for layer_name in layer_names])])


def set_precision(precision):
    if 'float16' in precision:
        if precision == 'mixed_float16':
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        else:
            tf.keras.backend.set_floatx('float16')