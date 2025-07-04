from typing import List, Optional

import keras
import numpy as np
import pytest

from airontools.constructors.layers import identity, layer_constructor


class TestLayerConstructor:

    @pytest.mark.parametrize("units", [[10, 20]])
    @pytest.mark.parametrize("length", [None, 3])
    @pytest.mark.parametrize("return_sequences", [True, False])
    @pytest.mark.parametrize("dropout_rate", [0.0, 0.1])
    @pytest.mark.parametrize("normalization_type", [None, "ln", "bn"])
    def test_output_units(
        self,
        units: List[int],
        length: int,
        return_sequences: bool,
        dropout_rate: float,
        normalization_type: Optional[str],
    ):
        if length is not None:
            shape = (length, units[0])
        else:
            shape = (units[0],)
        layer = keras.layers.Input(shape=shape)
        for hidden_layer_i, n_units in enumerate(units):
            layer = layer_constructor(
                x=layer,
                units=n_units,
                return_sequences=return_sequences,
                dropout_rate=dropout_rate,
                normalization_type=normalization_type,
            )
            assert layer.shape[0] is None
            assert layer.shape[-1] == n_units
            if length is not None and return_sequences:
                assert layer.shape[-2] == length


def test_identity():
    values = np.ones((10, 1))
    assert all(values == identity(values))
