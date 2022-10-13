from airontools.constructors.blocks import block_constructor


class TestBlockConstructor:

    def test_dense_units(self):
        units = [10, 5, 2]
        sub_units = [None, 10, 10, 5, 5, 2, 2]
        block = block_constructor(
            units=units,
            input_shape=(10,)
        )
        for n_units, layer in zip(sub_units, block.layers):
            if n_units is not None:
                assert n_units == layer.output_shape[-1]
