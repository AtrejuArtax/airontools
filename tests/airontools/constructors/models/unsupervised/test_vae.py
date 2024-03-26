import os
import tempfile

import keras
import numpy as np
import tensorflow as tf

from airontools.constructors.models.model import Model
from airontools.constructors.models.unsupervised.vae import VAE
from tests.airontools.constructors.example_data import LOSS_TOLERANCE, TABULAR_DATA


class TestVAE:
    model = VAE(
        input_shape=TABULAR_DATA.shape[1:],
        latent_dim=3,
    )
    assert isinstance(model, Model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    def test_fit(self):
        before_evaluation = self.model.evaluate(
            TABULAR_DATA,
        )["loss"]
        self.model.fit(
            TABULAR_DATA,
            epochs=10,
        )
        after_evaluation = self.model.evaluate(
            TABULAR_DATA,
        )["loss"]
        assert before_evaluation > after_evaluation

    def test_predict(self):
        prediction = self.model.predict(TABULAR_DATA)
        assert prediction.shape == TABULAR_DATA.shape

    def test_save_load_weights(self):
        before_weights_file_name = os.sep.join(
            [tempfile.gettempdir(), "before_weights"]
        )
        after_weights_file_name = os.sep.join([tempfile.gettempdir(), "after_weights"])
        before_evaluation = self.model.evaluate(
            TABULAR_DATA,
        )["loss"]
        self.model.save_weights(before_weights_file_name)
        self.model.fit(
            TABULAR_DATA,
            epochs=15,
        )
        after_evaluation = self.model.evaluate(
            TABULAR_DATA,
        )["loss"]
        assert before_evaluation > after_evaluation
        self.model.save_weights(after_weights_file_name)
        self.model.load_weights(before_weights_file_name)
        loss_difference = np.absolute(
            before_evaluation
            - self.model.evaluate(
                TABULAR_DATA,
            )["loss"]
        )
        assert loss_difference / before_evaluation <= LOSS_TOLERANCE
        self.model.load_weights(after_weights_file_name)
        loss_difference = np.absolute(
            after_evaluation
            - self.model.evaluate(
                TABULAR_DATA,
            )["loss"]
        )
        assert loss_difference / after_evaluation <= LOSS_TOLERANCE
