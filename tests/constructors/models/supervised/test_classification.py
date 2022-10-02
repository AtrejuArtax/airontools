import os

import numpy as np

from airontools.constructors.models.supervised.classification import ImageClassifierNN
from airontools.constructors.models.model import Model
from tensorflow.keras.optimizers import Adam
from numpy.random import normal
import tempfile


class TestImageClassifierNN:
    n_half_samples = 50
    n_classes = 2
    img_dim = 28
    data = np.concatenate([
        normal(loc=0.5, scale=1, size=(n_half_samples, img_dim, img_dim)),
        normal(loc=-0.5, scale=1, size=(n_half_samples, img_dim, img_dim))
    ])
    targets = np.concatenate([
        [[1, 0]] * n_half_samples,
        [[0, 1]] * n_half_samples,
    ])
    specs = dict(
        filters=32,
        kernel_size=15,
        strides=2,
        sequential_axis=-1,
        num_heads=3,
    )
    model = ImageClassifierNN(
        input_shape=tuple(data.shape[1:]),
        n_classes=n_classes,
        **specs,
    )
    assert isinstance(model, Model)
    assert not model._is_compiled
    model.compile(optimizer=Adam(learning_rate=0.001))
    assert model._is_compiled

    def test_fit(self):
        before_evaluation = self.model.evaluate(
            self.data,
            self.targets,
        )["loss"]
        self.model.fit(
            self.data,
            self.targets,
            epochs=5,
        )
        after_evaluation = self.model.evaluate(
            self.data,
            self.targets,
        )["loss"]
        assert before_evaluation > after_evaluation

    def test_predict(self):
        prediction = self.model.predict(self.data)
        assert prediction.shape == self.targets.shape

    def test_save_load_weights(self):
        before_weights_file_name = os.sep.join([tempfile.gettempdir(), "before_weights"])
        after_weights_file_name = os.sep.join([tempfile.gettempdir(), "after_weights"])
        before_evaluation = self.model.evaluate(
            self.data,
            self.targets,
        )["loss"]
        self.model.save_weights(before_weights_file_name)
        self.model.fit(
            self.data,
            self.targets,
            epochs=5,
        )
        after_evaluation = self.model.evaluate(
            self.data,
            self.targets,
        )["loss"]
        assert before_evaluation > after_evaluation
        self.model.save_weights(after_weights_file_name)
        self.model.load_weights(before_weights_file_name)
        assert before_evaluation == self.model.evaluate(
            self.data,
            self.targets,
        )["loss"]
        self.model.load_weights(after_weights_file_name)
        assert after_evaluation == self.model.evaluate(
            self.data,
            self.targets,
        )["loss"]