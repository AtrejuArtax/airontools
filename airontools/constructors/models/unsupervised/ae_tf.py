import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import binary_crossentropy
import json
import numpy as np
from airontools.constructors.layers import layer_constructor, identity


class ImageAE(Model):

    def __init__(self, latent_dim, **kwargs):
        super(ImageAE, self).__init__(**kwargs)

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")

        # Encoder
        encoder_inputs = Input(shape=(28, 28, 1))
        encoder_conv = layer_constructor(
            encoder_inputs,
            name='encoder_conv',
            filters=32,  # Number of filters used for the convolutional layer
            kernel_size=3,  # Kernel size used for the convolutional layer
            strides=2,  # Strides used for the convolutional layer
            sequential_axis=-1,  # It's the channel axis, used to define the sequence
            # for the self-attention layer
            num_heads=2,  # Self-attention heads applied after the convolutional layer
            units=latent_dim,  # Dense units applied after the self-attention layer
            advanced_reg=True)
        encoder_conv = layer_constructor(
            encoder_conv,
            name='z',
            units=latent_dim,
            advanced_reg=True)
        self.encoder = Model(encoder_inputs, encoder_conv, name='encoder')
        self.inputs = self.encoder.inputs

        # Z
        z_inputs = Input(shape=(latent_dim,))
        z = Lambda(identity, name='z')(z_inputs)
        self.z = Model(z_inputs, z, name='z')

        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        decoder_outputs = layer_constructor(
            latent_inputs,
            name='encoder_dense',
            units=7 * 7 * 64,
            advanced_reg=True)
        decoder_outputs = Reshape((7, 7, 64))(decoder_outputs)
        for i, filters, activation in zip([1, 2], [64, 32], ['relu', 'relu']):
            decoder_outputs = layer_constructor(
                decoder_outputs,
                name='decoder_conv',
                name_ext=str(i),
                filters=filters,
                kernel_size=3,
                strides=2,
                padding='same',
                conv_transpose=True,
                activation=activation,
                advanced_reg=True)
        decoder_outputs = layer_constructor(
            decoder_outputs,
            name='decoder_output',
            filters=1,
            kernel_size=3,
            padding='same',
            conv_transpose=True,
            activation='sigmoid',
            advanced_reg=True)
        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, data):
        total_loss, reconstruction_loss, tape = self.loss_evaluation(data, return_tape=True)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }

    def evaluate(self, data):
        total_loss, reconstruction_loss = self.loss_evaluation(data)
        return {
            'total_loss': total_loss.numpy(),
            'reconstruction_loss': reconstruction_loss.numpy(),
        }

    def loss_evaluation(self, data, return_tape=False):
        def loss_evaluation_():
            encoder = self.encoder(data)
            z = self.z(encoder)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            total_loss = reconstruction_loss
            return total_loss, reconstruction_loss

        if return_tape:
            with tf.GradientTape() as tape:
                total_loss, reconstruction_loss = loss_evaluation_()
            return total_loss, reconstruction_loss, tape
        else:
            total_loss, reconstruction_loss = loss_evaluation_()
            return total_loss, reconstruction_loss

    def save_weights(self, path):
        with open(path + '_encoder', 'w') as f:
            json.dump([w.tolist() for w in self.encoder.get_weights()], f)
        with open(path + '_decoder', 'w') as f:
            json.dump([w.tolist() for w in self.decoder.get_weights()], f)

    def load_weights(self, path):
        with open(path + '_encoder', 'r') as f:
            encoder_weights = [np.array(w) for w in json.load(f)]
        self.encoder.set_weights(encoder_weights)
        with open(path + '_decoder', 'r') as f:
            decoder_weights = [np.array(w) for w in json.load(f)]
        self.decoder.set_weights(decoder_weights)

    def summary(self):
        self.encoder.summary()
        self.z.summary()
        self.decoder.summary()
