from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model as KModel

from airontools.constructors.layers import layer_constructor
from airontools.constructors.models.model import Model
from airontools.on_the_fly import HyperDesignDropoutRate


class ImageClassifierNN(Model, KModel):
    def __init__(
        self,
        input_shape: Tuple[int],
        n_classes: int,
        model_name: str = "ImageClassifierNN",
        activation: str = "softmax",
        **kwargs,
    ):
        Model.__init__(self)
        KModel.__init__(self)

        # Loss tracker
        self.loss_tracker = Mean(name="loss")

        # Encoder
        encoder_inputs = Input(shape=input_shape)
        self._model = layer_constructor(
            encoder_inputs,
            input_shape=input_shape,
            units=n_classes,
            name=f"{model_name}_encoder",
            activation=activation,
            **kwargs,
        )
        self._model = KModel(
            inputs=encoder_inputs,
            outputs=self._model,
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

    def evaluate(self, x: np.array, y: np.array, **kwargs) -> Dict[str, float]:
        if "sample_weight_val" in kwargs.keys():
            sample_weight = kwargs["sample_weight_val"]
        else:
            sample_weight = 1
        return {"loss": self._loss_evaluation(y, self._model(x), sample_weight)}

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

    def train_step(self, data, **kwargs) -> dict:
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = 1
        with tf.GradientTape() as tape:
            loss = self._loss_evaluation(y, self._model(x), sample_weight)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _loss_evaluation(self, y, y_pred, sample_weight):
        loss = categorical_crossentropy(y, y_pred) * sample_weight
        loss = tf.reduce_mean(tf.reduce_sum(loss))
        return loss

    def call(self, inputs) -> None:
        """Call model."""
        return self._model(inputs)

    def summary(self) -> None:
        """Model summary."""
        self._model.summary()
