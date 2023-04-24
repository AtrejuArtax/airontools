import warnings

import tensorflow as tf


class HyperDesignDropoutRate:
    def __init__(self, model: tf.keras.models.Model, down=-0.01, up=0.01):
        self.rates = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.models.Model):
                for sub_layer in layer.layers:
                    self.__append_rate(sub_layer)
            elif isinstance(layer, tf.keras.layers.Layer):
                self.__append_rate(layer)
        self.actions_space = {}
        for action_name, action_value in zip(["down", "stay", "up"], [down, 0.0, up]):
            self.actions_space.update(
                {
                    action_name: tf.constant(
                        action_value,
                        dtype=model.dtype,
                        name="_".join(["rate", action_name]),
                    ),
                },
            )

    def __append_rate(self, layer: tf.keras.layers.Layer):
        if hasattr(layer, "rate"):
            if isinstance(layer.rate, tf.Variable):
                self.rates += [layer.rate]
            else:
                warnings.warn(
                    "layer {} does not contain a rate as a tf.Variable".format(
                        layer.name,
                    ),
                )

    def set_action(self, action: str):
        assert action in ["down", "stay", "up"]
        for rate in self.rates:
            new_rate = rate + self.actions_space[action]
            new_rate_ = tf.keras.backend.get_value(new_rate)
            if 0 <= new_rate_ < 1:
                tf.keras.backend.set_value(rate, new_rate)
