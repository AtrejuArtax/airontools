from tensorflow import Tensor
from tensorflow.keras.layers import Input

from airontools.constructors.layers import dropout_layer_constructor, layer_constructor


class TestLayerConstructor:
    def test_output_units(self):
        units = [10, 5, 2]
        input_layer = Input(shape=(units[0],))
        for n_units in units:
            layer = layer_constructor(x=input_layer, units=n_units)
            assert n_units == layer.shape[-1]


class TestDropoutLayerConstructor:
    def test_output_units(self):
        units = [10, 5, 2]
        input_layer = Input(shape=(units[0],))
        for n_units in units:
            layer = layer_constructor(x=input_layer, units=n_units)
            layer = dropout_layer_constructor(x=layer)
            assert n_units == layer.shape[-1]
