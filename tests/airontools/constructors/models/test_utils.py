import tensorflow as tf

from airontools.constructors.models.unsupervised.vae import VAE
from airontools.constructors.utils import get_latent_model
from tests.airontools.constructors.models.example_data import TABULAR_DATA


class TestGetLatentModel:
    model_name = "test_vae"
    model = VAE(
        model_name=model_name,
        input_shape=TABULAR_DATA.shape[1:],
        latent_dim=2,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam())

    def test_successful_get_latent_model(self):
        latent_model = get_latent_model(
            model=self.model,
            layer_name="_".join([self.model_name, "encoder"]),
        )
        assert isinstance(latent_model, tf.keras.models.Model)

    def test_unsuccessful_get_latent_model(self):
        latent_model = get_latent_model(
            model=self.model,
            layer_name="_".join([self.model_name, "pepito"]),
        )
        assert latent_model is None
