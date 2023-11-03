import json
from typing import Dict, Tuple, Union

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from airontools.constructors.layers import layer_constructor
from airontools.constructors.models.model import Model
from airontools.on_the_fly.hyper_design_dropout_rate import HyperDesignDropoutRate


class AE(Model, tf.keras.models.Model):
    """AutoEncoder model."""

    def __init__(
        self,
        input_shape: Tuple[int],
        model_name: str = "AE",
        output_activation: str = "linear",
        latent_dim: int = 3,
        **kwargs,
    ):
        Model.__init__(self)
        tf.keras.models.Model.__init__(self)

        # Loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # Encoder
        encoder_inputs, self.encoder = self._get_encoder_model(
            input_shape=input_shape,
            latent_dim=latent_dim,
            model_name=model_name,
            **kwargs,
        )

        # Z
        self.z = self._get_z_model(
            latent_dim=latent_dim,
            model_name=model_name,
            **kwargs,
        )

        # Decoder
        self.decoder = self._get_decoder_model(
            units=self.encoder.input_shape[-1],
            latent_dim=latent_dim,
            model_name=model_name,
            output_activation=output_activation,
            **kwargs,
        )

        # AE
        self._model = tf.keras.models.Model(
            inputs=encoder_inputs,
            outputs=self.decoder(self.z(self.encoder(encoder_inputs))),
            name=model_name,
        )

        # Hyper design on the fly
        self.hyper_design_dropout_rate = HyperDesignDropoutRate(model=self._model)

    def compile(self, *args, **kwargs) -> None:
        """Compile model."""
        tf.keras.models.Model.compile(self, *args, **kwargs)

    def fit(self, *args, **kwargs) -> None:
        """Fit."""
        tf.keras.models.Model.fit(self, *args, **kwargs)

    def evaluate(
        self, x: Union[NDArray[float], tf.Tensor], **kwargs
    ) -> Dict[str, tf.Tensor]:
        """Evaluate model."""
        reconstructed = self._model(x)
        loss = self._loss_evaluation(reconstructed, x)
        return {"loss": loss}

    def predict(self, *args, **kwargs) -> NDArray[float]:
        """Predict."""
        return tf.keras.models.Model.predict(self, *args, **kwargs)

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
        reconstructed = self._model(inputs)
        self.add_loss(self._loss_evaluation(reconstructed, inputs))
        return reconstructed

    def summary(self, **kwargs) -> None:
        """Model summary."""
        self._model.summary(**kwargs)

    def _loss_evaluation(
        self, reconstructed: tf.Tensor, inputs: tf.Tensor
    ) -> tf.Tensor:
        rec_loss = tf.reduce_mean((inputs - reconstructed) ** 2)
        return rec_loss

    @staticmethod
    def _get_encoder_model(
        input_shape: Tuple[int], latent_dim: int, model_name: str, **kwargs
    ) -> Tuple[tf.keras.layers.Input, tf.keras.models.Model]:
        encoder_inputs = tf.keras.layers.Input(shape=input_shape)
        encoder = layer_constructor(
            encoder_inputs,
            units=latent_dim,
            name=f"{model_name}_encoder",
            **kwargs,
        )
        encoder = tf.keras.models.Model(
            inputs=encoder_inputs,
            outputs=encoder,
            name=f"{model_name}_encoder",
        )
        return encoder_inputs, encoder

    @staticmethod
    def _get_z_model(
        latent_dim: int, model_name: str, **kwargs
    ) -> tf.keras.models.Model:
        z_inputs = tf.keras.layers.Input(shape=(latent_dim,))
        z = layer_constructor(
            z_inputs,
            units=latent_dim,
            name=f"{model_name}_z",
            **kwargs,
        )
        z = tf.keras.models.Model(
            inputs=z_inputs,
            outputs=z,
            name=f"{model_name}_z",
        )
        return z

    @staticmethod
    def _get_decoder_model(
        units: int, latent_dim: int, model_name: str, output_activation: str, **kwargs
    ) -> tf.keras.models.Model:
        decoder_inputs = tf.keras.layers.Input(shape=(latent_dim,))
        decoder = layer_constructor(
            decoder_inputs,
            units=units,
            name=f"{model_name}_decoder",
            activation=output_activation,
            **kwargs,
        )
        decoder = tf.keras.models.Model(
            inputs=decoder_inputs,
            outputs=decoder,
            name=f"{model_name}_decoder",
        )
        return decoder
