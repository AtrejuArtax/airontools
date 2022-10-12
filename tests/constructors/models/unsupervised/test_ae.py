import os
import tempfile

from tensorflow.keras.optimizers import Adam

from airontools.constructors.models.model import Model
from airontools.constructors.models.unsupervised.ae import AE
from tests.constructors.models.example_data import TABULAR_DATA


class TestAE:
    model = AE(
        input_shape=TABULAR_DATA.shape[1:],
        latent_dim=3,
    )
    assert isinstance(model, Model)
    assert not model._is_compiled
    model.compile(optimizer=Adam(learning_rate=0.001))
    assert model._is_compiled

    def test_fit(self):
        before_evaluation = self.model.evaluate(
            TABULAR_DATA,
        )["loss"]
        self.model.fit(
            TABULAR_DATA,
            epochs=5,
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
            epochs=5,
        )
        after_evaluation = self.model.evaluate(
            TABULAR_DATA,
        )["loss"]
        assert before_evaluation > after_evaluation
        self.model.save_weights(after_weights_file_name)
        self.model.load_weights(before_weights_file_name)
        assert (
            before_evaluation
            == self.model.evaluate(
                TABULAR_DATA,
            )["loss"]
        )
        self.model.load_weights(after_weights_file_name)
        assert (
            after_evaluation
            == self.model.evaluate(
                TABULAR_DATA,
            )["loss"]
        )
