from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model as KModel

from airontools.constructors.layers import layer_constructor
from airontools.constructors.models.model import Model
from airontools.on_the_fly import HyperDesignDropoutRate


class AE(Model, KModel):
    def __init__(
        self,
        input_shape: Tuple[int],
        model_name: str = "AE",
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
        self.encoder = KModel(
            inputs=encoder_inputs,
            outputs=self.encoder,
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
        reconstructed = self._model(x)
        return {"loss": self._loss_evaluation(reconstructed, x)}

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
        reconstructed = self._model(inputs)
        self.add_loss(self._loss_evaluation(inputs, reconstructed))
        return reconstructed

    def summary(self) -> None:
        """Model summary."""
        self._model.summary()

    def _loss_evaluation(self, reconstructed, inputs, **kwargs):
        rec_loss = tf.reduce_mean((inputs - reconstructed) ** 2)
        return rec_loss
