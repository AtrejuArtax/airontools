import json
import os.path

import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean
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
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            kernel_regularizer_l1=kernel_regularizer_l1,
            kernel_regularizer_l2=kernel_regularizer_l2,
            bias_regularizer_l1=bias_regularizer_l1,
            bias_regularizer_l2=bias_regularizer_l2,
            bn=bn,
        )
        self.loss_tracker = Mean(name="loss")
        self.classification_loss_tracker = Mean(name="classification_loss")
        self.cce = CategoricalCrossentropy()

        # Encoder
        encoder_inputs = Input(shape=input_shape)
        encoder_inputs = Lambda(divide_by_255)(encoder_inputs)
        encoder = layer_constructor(
            encoder_inputs,
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

    def call(self, inputs):
        return self.encoder(inputs)

    def save_weights(self, path):
        with open(path + "_weights", "w") as f:
            json.dump([w.tolist() for w in self.encoder.get_weights()], f)

    def load_weights(self, path):
        with open(path + "_weights", "r") as f:
            encoder_weights = [np.array(w) for w in json.load(f)]
        self.encoder.set_weights(encoder_weights)

    def summary(self):
        self.encoder.summary()


def divide_by_255(x):
    return x / 255
