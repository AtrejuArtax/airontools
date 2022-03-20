import tensorflow as tf
import tensorflow.keras.backend as k_bcknd
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import warnings


class HyperDesignDropoutRate(object):
    def __init__(self, model: Model, down=-0.01, up=0.01):
        self.rates = []
        for layer in model.layers:
            if isinstance(layer, Model):
                for sub_layer in layer.layers:
                    self.__append_rate(sub_layer)
            elif isinstance(layer, Layer):
                self.__append_rate(layer)
        self.actions_space = dict(
            down=tf.constant(down, dtype=model.dtype, name="rate_down"),
            stay=0,
            up=tf.constant(up, dtype=model.dtype, name="rate_up"),
        )

    def __append_rate(self, layer: Layer):
        if hasattr(layer, "rate"):
            if isinstance(layer.rate, tf.Variable):
                self.rates += [layer.rate]
            else:
                warnings.warn("layer {} does not contain a rate as a tf.Variable".format(layer.name))

    def set_action(self, action: str):
        assert action in ["down", "stay", "up"]
        for rate in self.rates:
            new_rate = rate + self.actions_space[action]
            new_rate_ = k_bcknd.get_value(new_rate)
            if 0 <= new_rate_ < 1:
                k_bcknd.set_value(rate, new_rate)
