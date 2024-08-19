import json
from typing import Dict, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from airontools.constructors.layers import layer_constructor
from airontools.constructors.models.model import Model
from airontools.on_the_fly.hyper_design_dropout_rate import HyperDesignDropoutRate


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, **kwargs) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model, keras.models.Model):
    """Variational AutoEncoder model.
    For more information refer to this paper: https://arxiv.org/pdf/1312.6114.pdf"""

    def __init__(
        self,
        input_shape: Tuple[int],
        model_name: str = "VAE",
        output_activation: str = "linear",
        latent_dim: int = 3,
        **kwargs,
    ):
        Model.__init__(self)
        keras.models.Model.__init__(self)

        # Loss tracker
        self.loss_tracker = keras.metrics.Mean(name="loss")

        # Encoder
        encoder_inputs = keras.layers.Input(shape=input_shape)
        encoder = layer_constructor(
            encoder_inputs,
            units=latent_dim,
            name=f"{model_name}_encoder",
            **kwargs,
        )
        self.z_mean = layer_constructor(
            encoder,
            units=latent_dim,
            name=f"{model_name}_z_mean",
            **kwargs,
        )
        self.z_log_var = layer_constructor(
            encoder,
            units=latent_dim,
            name=f"{model_name}_z_log_var",
            **kwargs,
        )
        self.encoder = keras.models.Model(
            inputs=encoder_inputs,
            outputs=[self.z_mean, self.z_log_var],
            name=f"{model_name}_encoder",
        )

        # Z
        z_mean = keras.layers.Input(shape=(latent_dim,))
        z_log_var = keras.layers.Input(shape=(latent_dim,))
        z = Sampling(name=f"{model_name}_sampling")([z_mean, z_log_var])
        self.z = keras.models.Model(
            inputs=[z_mean, z_log_var],
            outputs=z,
            name=f"{model_name}_z",
        )

        # Decoder
        decoder_inputs = keras.layers.Input(shape=(latent_dim,))
        decoder = layer_constructor(
            decoder_inputs,
            units=input_shape[0],
            name=f"{model_name}_decoder",
            activation=output_activation,
            **kwargs,
        )
        self.decoder = keras.models.Model(
            inputs=decoder_inputs,
            outputs=decoder,
            name=f"{model_name}_decoder",
        )
        self.encoder(encoder_inputs)
        self.z(self.encoder(encoder_inputs))
        self.decoder(self.z(self.encoder(encoder_inputs)))

        # AE
        self._model = keras.models.Model(
            inputs=encoder_inputs,
            outputs=self.decoder(self.z(self.encoder(encoder_inputs))),
            name=model_name,
        )

        # Hyper design on the fly
        self.hyper_design_dropout_rate = HyperDesignDropoutRate(model=self._model)

    def compile(self, *args, **kwargs) -> None:
        """Compile model."""
        keras.models.Model.compile(self, *args, **kwargs)

    def fit(self, *args, **kwargs) -> None:
        """Fit."""
        keras.models.Model.fit(self, *args, **kwargs)

    def evaluate(
        self, x: Union[NDArray[float], tf.Tensor], **kwargs
    ) -> Dict[str, tf.Tensor]:
        """Evaluate model."""
        z_mean, z_log_var = self.encoder(x)
        z = self.z([z_mean, z_log_var])
        loss = self._loss_evaluation(
            inputs=x,
            z_mean=z_mean,
            z_log_var=z_log_var,
            z=z,
        )
        return {"loss": loss}

    def predict(self, *args, **kwargs) -> NDArray[float]:
        """Predict."""
        return keras.models.Model.predict(self, *args, **kwargs)

    def save_weights(self, path: str) -> None:
        """Save model weights."""
        with open(path + "_weights", "w") as f:
            json.dump([w.tolist() for w in self._model.get_weights()], f)

    def load_weights(self, path: str) -> None:
        """Load model weights."""
        with open(path + "_weights") as f:
            encoder_weights = [np.array(w) for w in json.load(f)]
        self._model.set_weights(encoder_weights)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """Call model."""
        z_mean, z_log_var = self.encoder(inputs)
        z = self.z([z_mean, z_log_var])
        loss, reconstructed = self._loss_evaluation(
            inputs=inputs,
            z_mean=z_mean,
            z_log_var=z_log_var,
            z=z,
            return_reconstruction=True,
        )
        self.add_loss(loss)
        return reconstructed

    def summary(self, **kwargs) -> None:
        """Model summary."""
        self._model.summary(**kwargs)

    def _loss_evaluation(
        self,
        inputs: tf.Tensor,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
        z: tf.Tensor,
        return_reconstruction: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        reconstructed = self.decoder(z)
        rec_loss = tf.reduce_mean((inputs - reconstructed) ** 2)
        # Add KL divergence regularization loss.
        kld_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        if return_reconstruction:
            return rec_loss + 0.1 * kld_loss, reconstructed
        else:
            return rec_loss + 0.1 * kld_loss
