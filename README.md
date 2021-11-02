# AIronTools

AIronTools (Beta) is a Python library that provides the user with deep learning tools built to work with 
[tensorflow](https://github.com/tensorflow/tensorflow) (or [pytorch](https://github.com/pytorch/pytorch) in the future) 
as a backend.

Key features:

1. Model constructor that allows multiple models to be optimized in parallel across multiple GPUs. 
2. Block constructor to build customised blocks/models.
3. Layer constructor to build customised layers.
4. Preprocessing utils.
   
### Installation

`pip install airontools`

### Custom Keras subclass to build a variational autoencoder (VAE) with airontools

``` python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Reshape
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from airontools.preprocessing import train_val_split
from airontools.model_constructors import layer_constructor

class VAE(Model):
    def __init__(self, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

        # Encoder
        encoder_inputs = Input(shape=(28, 28, 1))
        encoder_conv = layer_constructor(
            encoder_inputs, name='encoder_conv', filters=32, kernel_size=3, strides=2, advanced_reg=True)
        z_mean = layer_constructor(encoder_conv, name='encoder_mean', units=latent_dim, advanced_reg=True)
        z_log_var = layer_constructor(encoder_conv, name='encoder_log_var', units=latent_dim, advanced_reg=True)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        decoder_outputs = layer_constructor(latent_inputs, name='encoder_dense', units=7 * 7 * 64, advanced_reg=True)
        decoder_outputs = Reshape((7, 7, 64))(decoder_outputs)
        for i, filters, activation in zip([1, 2], [64, 32], ['relu', 'relu']):
            decoder_outputs = layer_constructor(
                decoder_outputs,
                name='decoder_conv', name_ext=str(i), activation=activation, filters=filters, kernel_size=3,
                strides=2, padding='same', conv_transpose=True, advanced_reg=True)
        decoder_outputs = layer_constructor(
            decoder_outputs,
            name='decoder_output', activation='sigmoid', filters=1, kernel_size=3, padding='same',
            conv_transpose=True, advanced_reg=True)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")
```

### More examples

see usage examples in [aironsuit/examples](https://github.com/AtrejuArtax/aironsuit/tree/master/examples)