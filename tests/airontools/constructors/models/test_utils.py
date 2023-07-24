import tensorflow as tf

from airontools.constructors.models.unsupervised.vae import VAE
from airontools.constructors.utils import (
    get_latent_model,
    get_regularizer,
    set_precision,
)
from tests.airontools.constructors.models.example_data import TABULAR_DATA


class TestGetLatentModel:
    model_name = "test_vae"
    model = VAE(
        model_name=model_name,
        input_shape=TABULAR_DATA.shape[1:],
        latent_dim=2,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam())

    def test_successful_case(self):
        latent_model = get_latent_model(
            model=self.model,
            layer_name="_".join([self.model_name, "encoder"]),
        )
        assert isinstance(latent_model, tf.keras.models.Model)

    def test_unsuccessful_case(self):
        latent_model = get_latent_model(
            model=self.model,
            layer_name="_".join([self.model_name, "pepito"]),
        )
        assert latent_model is None


class TestGetRegularizer:
    def test_l1_case(self):
        regularizer = get_regularizer(l1_value=0.001)
        assert isinstance(regularizer, tf.keras.regularizers.Regularizer)

    def test_l2_case(self):
        regularizer = get_regularizer(l2_value=0.001)
        assert isinstance(regularizer, tf.keras.regularizers.Regularizer)

    def test_l1_l2_case(self):
        regularizer = get_regularizer(l1_value=0.001, l2_value=0.001)
        assert isinstance(regularizer, tf.keras.regularizers.Regularizer)

    def test_none_case(self):
        regularizer = get_regularizer()
        assert regularizer is None
