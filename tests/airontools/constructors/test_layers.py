import keras
import numpy as np

from airontools.constructors.layers import (
    CustomDropout,
    dropout_layer_constructor,
    identity,
    layer_constructor,
)


class TestLayerConstructor:
    def test_output_units(self):
        units = [10, 5, 2]
        input_layer = keras.layers.Input(shape=(units[0],))
        for n_units in units:
            layer = layer_constructor(x=input_layer, units=n_units)
            assert n_units == layer.shape[-1]


class TestDropoutLayerConstructor:
    def test_output_units(self):
        units = [10, 5, 2]
        input_layer = keras.layers.Input(shape=(units[0],))
        for n_units in units:
            layer = layer_constructor(x=input_layer, units=n_units)
            layer = dropout_layer_constructor(x=layer, dropout_rate=0.1)
            assert n_units == layer.shape[-1]


def test_identity():
    values = np.ones((10, 1))
    assert all(values == identity(values))


class TestCustomDropout:
    def test_output_units(self):
        # Test output dimensions are the same as input dimensions.
        units = [10, 5, 2]
        for n_units in units:
            input_layer = keras.layers.Input(shape=(n_units,))
            layer = CustomDropout(rate=0.1)(input_layer)
            assert n_units == layer.shape[-1]
