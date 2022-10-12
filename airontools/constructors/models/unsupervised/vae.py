from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model as KModel
from airontools.constructors.layers import layer_constructor
from airontools.constructors.models.model import Model
from airontools.on_the_fly import HyperDesignDropoutRate


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model, KModel):
    def __init__(
        self,
        input_shape: Tuple[int],
        model_name: str = "VAE",
        output_activation: str = "softmax",
        latent_dim: int = 3,
        **kwargs,
    ):
        Model.__init__(self)
        KModel.__init__(self)

        # Loss tracker
        self.loss_tracker = Mean(name="loss")

        # Encoder
        encoder_inputs = Input(shape=input_shape)
        self.encoder = layer_constructor(
            encoder_inputs,
            input_shape=input_shape,
            units=latent_dim,
            name=f"{model_name}_encoder",
            **kwargs,
        )
        self.z_mean = layer_constructor(
            self.encoder,
            input_shape=(latent_dim,),
            units=latent_dim,
            name=f"{model_name}_z_mean",
            **kwargs,
        )
        self.z_log_var = layer_constructor(
            self.encoder,
            input_shape=(latent_dim,),
            units=latent_dim,
            name=f"{model_name}_z_log_var",
            **kwargs,
        )
        self.z = Sampling(name=f"{model_name}_z")([self.z_mean, self.z_log_var])
        self.encoder = KModel(
            inputs=encoder_inputs,
            outputs=[self.z_mean, self.z_log_var, self.z],
            name=f"{model_name}_encoder",
        )

        # Z
        z_inputs = Input(shape=(latent_dim,))
        self.z = layer_constructor(
            z_inputs,
            input_shape=(latent_dim,),
            units=latent_dim,
            name=f"{model_name}_z",
            **kwargs,
        )
        self.z = KModel(
            inputs=z_inputs,
            outputs=self.z,
            name=f"{model_name}_z",
        )

        # Decoder
        decoder_inputs = Input(shape=(latent_dim,))
        self.decoder = layer_constructor(
            decoder_inputs,
            input_shape=(latent_dim,),
            units=self.encoder.input_shape[-1],
            name=f"{model_name}_decoder",
            activation=output_activation,
            **kwargs,
        )
        self.decoder = KModel(
            inputs=decoder_inputs,
            outputs=self.decoder,
            name=f"{model_name}_decoder",
        )

        # AE
        self._model = KModel(
            inputs=encoder_inputs,
            outputs=self.decoder(self.z(self.encoder(encoder_inputs))),
            name=model_name,
        )

        # Hyper design on the fly
        self.hyper_design_dropout_rate = HyperDesignDropoutRate(model=self._model)

    def compile(self, *args, **kwargs) -> None:
        """Compile model."""
        KModel.compile(self, *args, **kwargs)

    def fit(self, *args, **kwargs) -> None:
        """Compile model."""
        KModel.fit(self, *args, **kwargs)

    def evaluate(self, x: np.array, **kwargs) -> Dict[str, float]:
        """Evaluate model."""
        z_mean, z_log_var, z = self.encoder(x)
        loss = self._loss_evaluation(
            inputs=x,
            z_mean=z_mean,
            z_log_var=z_log_var,
            z=z,
        )
        return {"loss": loss}

    def predict(self, *args, **kwargs) -> np.array:
        """Predict model."""
        return KModel.predict(self, *args, **kwargs)

    def save_weights(self, path: str) -> None:
        with open(path + "_weights", "w") as f:
            json.dump([w.tolist() for w in self._model.get_weights()], f)

    def load_weights(self, path: str) -> None:
        with open(path + "_weights") as f:
            encoder_weights = [np.array(w) for w in json.load(f)]
        self._model.set_weights(encoder_weights)

    def call(self, inputs) -> None:
        """Call model."""
        z_mean, z_log_var, z = self.encoder(inputs)
        loss, reconstructed = self._loss_evaluation(
            inputs=inputs,
            z_mean=z_mean,
            z_log_var=z_log_var,
            z=z,
            return_reconstruction=True,
        )
        self.add_loss(loss)
        return reconstructed

    def summary(self) -> None:
        """Model summary."""
        self._model.summary()

    def _loss_evaluation(self, inputs, z_mean, z_log_var, z, return_reconstruction=False, **kwargs):
        reconstructed = self.decoder(z)
        rec_loss = tf.reduce_mean(
            (inputs - reconstructed) ** 2
        )
        # Add KL divergence regularization loss.
        kld_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        if return_reconstruction:
            return rec_loss + 0.1 * kld_loss, reconstructed
        else:
            return rec_loss + 0.1 * kld_loss
