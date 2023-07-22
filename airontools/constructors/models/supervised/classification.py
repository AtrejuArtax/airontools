import json
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from airontools.constructors.layers import layer_constructor
from airontools.constructors.models.model import Model
from airontools.on_the_fly.hyper_design_dropout_rate import HyperDesignDropoutRate


class ImageClassifierNN(Model, tf.keras.models.Model):
    """Image classifier model."""

    def __init__(
        self,
        input_shape: Tuple[int],
        n_classes: int,
        model_name: str = "ImageClassifierNN",
        output_activation: str = "softmax",
        **kwargs,
    ):
        Model.__init__(self)
        tf.keras.models.Model.__init__(self)

        # Loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=input_shape)
        encoder_outputs = layer_constructor(
            x=encoder_inputs,
            units=n_classes,
            name=f"{model_name}_encoder",
            activation=output_activation,
            **kwargs,
        )
        self._model = tf.keras.models.Model(
            inputs=encoder_inputs,
            outputs=encoder_outputs,
            name=model_name,
        )

        # Hyper design on the fly
        self.hyper_design_dropout_rate = HyperDesignDropoutRate(model=self._model)

    def compile(self, *args, **kwargs) -> None:
        """Compile model."""
        tf.keras.models.Model.compile(self, *args, **kwargs)

    def fit(self, *args, **kwargs) -> None:
        """Fit model."""
        tf.keras.models.Model.fit(self, *args, **kwargs)

    def evaluate(
        self, x: NDArray[float], y: NDArray[float], **kwargs
    ) -> Dict[str, tf.Tensor]:
        """Evaluate model."""
        if "sample_weight_val" in kwargs.keys():
            sample_weight = kwargs["sample_weight_val"]
        else:
            sample_weight = 1
        return {"loss": self._loss_evaluation(y, self._model(x), sample_weight)}

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

    def train_step(self, data: List[NDArray[float]], **kwargs) -> Dict[str, float]:
        """Train step."""
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

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """Call model."""
        return self._model(inputs, **kwargs)

    def summary(self, **kwargs) -> None:
        """Model summary."""
        self._model.summary(**kwargs)

    def _loss_evaluation(
        self, y: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor
    ) -> tf.Tensor:
        loss = tf.keras.metrics.categorical_crossentropy(y, y_pred) * sample_weight
        loss = tf.reduce_mean(tf.reduce_sum(loss))
        return loss
