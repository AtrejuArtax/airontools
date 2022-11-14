import numpy as np
from numpy.random import normal
from tensorflow.keras.optimizers import Adam

from airontools.constructors.models.unsupervised.vae import VAE

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
model.compile(optimizer=Adam(learning_rate=0.001))
model.fit(
    tabular_data,
    epochs=10,
)
print("VAE evaluation:", float(model.evaluate(tabular_data)["loss"]))