import json
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from airontools.constructors.layers import layer_constructor
from airontools.constructors.models.model import Model
from airontools.on_the_fly import HyperDesignDropoutRate


class AE(Model, tf.keras.models.Model):
    """AutoEncoder model."""

    def __init__(
        self,
        input_shape: Tuple[int],
        model_name: str = "AE",
        output_activation: str = "softmax",
        latent_dim: int = 3,
        **kwargs,
    ):
        Model.__init__(self)
        tf.keras.models.Model.__init__(self)

        # Loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=input_shape)
        self.encoder = layer_constructor(
            encoder_inputs,
            units=latent_dim,
            name=f"{model_name}_encoder",
            **kwargs,
        )
        self.encoder = tf.keras.models.Model(
            inputs=encoder_inputs,
            outputs=self.encoder,
            name=f"{model_name}_encoder",
        )

        # Z
        z_inputs = tf.keras.layers.Input(shape=(latent_dim,))
        self.z = layer_constructor(
            z_inputs,
            units=latent_dim,
            name=f"{model_name}_z",
            **kwargs,
        )
        self.z = tf.keras.models.Model(
            inputs=z_inputs,
            outputs=self.z,
            name=f"{model_name}_z",
        )

        # Decoder
        decoder_inputs = tf.keras.layers.Input(shape=(latent_dim,))
        self.decoder = layer_constructor(
            decoder_inputs,
            units=self.encoder.input_shape[-1],
            name=f"{model_name}_decoder",
            activation=output_activation,
            **kwargs,
        )
        self.decoder = tf.keras.models.Model(
            inputs=decoder_inputs,
            outputs=self.decoder,
            name=f"{model_name}_decoder",
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
        """Compile model."""
        tf.keras.models.Model.fit(self, *args, **kwargs)

    def evaluate(self, x: NDArray[float], **kwargs) -> Dict[str, tf.Tensor]:
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
