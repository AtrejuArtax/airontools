import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model

from airontools.constructors.layers import layer_constructor


class ImageClassifierNN(Model):
    def __init__(
        self,
        input_shape,
        dropout_rate,
        kernel_regularizer_l1,
        kernel_regularizer_l2,
        bias_regularizer_l1,
        bias_regularizer_l2,
        bn,
        **kwargs
    ):
        super(ImageClassifierNN, self).__init__(**kwargs)

        reg_kwargs_ = dict(
            dropout_rate=dropout_rate,
            kernel_regularizer_l1=kernel_regularizer_l1,
            kernel_regularizer_l2=kernel_regularizer_l2,
            bias_regularizer_l1=bias_regularizer_l1,
            bias_regularizer_l2=bias_regularizer_l2,
            bn=bn,
        )

        self.loss_tracker = Mean(name="loss")

        # Encoder
        encoder_inputs = Input(shape=input_shape)
        encoder = layer_constructor(
            encoder_inputs,
            input_shape=input_shape,
            name="encoder_conv",
            filters=32,  # Number of filters used for the convolutional layer
            kernel_size=15,  # Kernel size used for the convolutional layer
            strides=2,  # Strides used for the convolutional layer
            sequential_axis=-1,  # It's the channel axis, used to define the sequence
            # for the self-attention layer
            num_heads=3,  # Self-attention heads applied after the convolutional layer
            units=10,  # Dense units applied after the self-attention layer
            activation="softmax",  # Output activation function
            **reg_kwargs_  # Regularization arguments
        )
        self.encoder = Model(encoder_inputs, encoder, name="encoder")

    @tf.function
    def train_step(self, data, **kwargs):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = 1
        with tf.GradientTape() as tape:
            loss = self._loss_evaluation(y, self.encoder(x), sample_weight)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @tf.function
    def evaluate(self, x, y, **kwargs):
        if "sample_weight_val" in kwargs.keys():
            sample_weight = kwargs["sample_weight_val"]
        else:
            sample_weight = 1
        return {"loss": self._loss_evaluation(y, self.encoder(x), sample_weight)}

    def _loss_evaluation(self, y, y_pred, sample_weight):
        loss = categorical_crossentropy(y, y_pred) * sample_weight
        loss = tf.reduce_mean(tf.reduce_sum(loss))
        return loss

    @tf.function
    def call(self, inputs):
        return self.encoder(inputs)

    def save_weights(self, path):
        with open(path + "_weights", "w") as f:
            json.dump([w.tolist() for w in self.encoder.get_weights()], f)

    def load_weights(self, path):
        with open(path + "_weights", "r") as f:
            encoder_weights = [np.array(w) for w in json.load(f)]
        self.encoder.set_weights(encoder_weights)

    @tf.function
    def summary(self):
        self.encoder.summary()
