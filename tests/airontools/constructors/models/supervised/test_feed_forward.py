import os
import tempfile

import keras

from airontools.constructors.models.model import Model
from airontools.constructors.models.supervised.feed_forward import FeedForward
from tests.airontools.constructors.example_data import IMG_DATA, N_CLASSES, TARGETS


class TestFeedForward:
    specs = dict(
        filters=32,
        kernel_size=15,
        strides=2,
        sequential_axis=-1,
        num_heads=3,
    )
    model = FeedForward(
        input_shape=tuple(IMG_DATA.shape[1:]),
        n_outputs=N_CLASSES,
        **specs,
    )
    assert isinstance(model, Model)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    def test_fit(self):
        before_evaluation = self.model.evaluate(
            IMG_DATA,
            TARGETS,
        )["loss"]
        self.model.fit(
            IMG_DATA,
            TARGETS,
            epochs=5,
        )
        after_evaluation = self.model.evaluate(
            IMG_DATA,
            TARGETS,
        )["loss"]
        assert before_evaluation > after_evaluation

    def test_predict(self):
        prediction = self.model.predict(IMG_DATA)
        assert prediction.shape == TARGETS.shape

    def test_save_load_weights(self):
        before_weights_file_name = os.sep.join(
            [tempfile.gettempdir(), "before_weights"]
        )
        after_weights_file_name = os.sep.join([tempfile.gettempdir(), "after_weights"])
        before_evaluation = self.model.evaluate(
            IMG_DATA,
            TARGETS,
        )["loss"]
        self.model.save_weights(before_weights_file_name)
        self.model.fit(
            IMG_DATA,
            TARGETS,
            epochs=5,
        )
        after_evaluation = self.model.evaluate(
            IMG_DATA,
            TARGETS,
        )["loss"]
        assert before_evaluation > after_evaluation
        self.model.save_weights(after_weights_file_name)
        self.model.load_weights(before_weights_file_name)
        assert (
            before_evaluation
            == self.model.evaluate(
                IMG_DATA,
                TARGETS,
            )["loss"]
        )
        self.model.load_weights(after_weights_file_name)
        assert (
            after_evaluation
            == self.model.evaluate(
                IMG_DATA,
                TARGETS,
            )["loss"]
        )
