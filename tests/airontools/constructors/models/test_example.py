import numpy as np
import tensorflow as tf
from numpy.random import normal

from airontools.constructors.models.unsupervised.vae import VAE


def test_example():
    tabular_data = np.concatenate(
        [
            normal(loc=0.5, scale=1, size=(100, 10)),
            normal(loc=-0.5, scale=1, size=(100, 10)),
        ]
    )
    model = VAE(
        input_shape=tabular_data.shape[1:],
        latent_dim=3,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.fit(
        tabular_data,
        epochs=10,
    )
    assert isinstance(model.evaluate(tabular_data)["loss"].numpy(), np.float32)
