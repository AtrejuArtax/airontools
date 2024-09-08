# AIronTools

AIronTools (Beta) is a Python library that provides the user with higher level state-of-the-art deep learning tools built to work with 
[tensorflow](https://github.com/tensorflow/tensorflow) as a backend. The main goal of this repository is to enable fast model
design for both POCs and production.

Key features:

1. Out-of-the-box models ready to be used.
2. Block constructor to build customised blocks/models.
3. Layer constructor to build customised layers such as sequential, convolutional, self-attention or dense, and combinations of them.
4. Preprocessing tools.
5. On the fly non-topological hyper-parameter optimization. For now only the dropout regularization is compatible with this feature, in the future others such as l1 and l2 regularization will be compatible too.
6. Latent representations for visualization purposes.
   
### Installation

`pip install airontools`

### Custom Keras subclass to build a variational autoencoder (VAE) with airontools and compatible with [aironsuit](https://github.com/AtrejuArtax/aironsuit/)

``` python
import numpy as np
import tensorflow as tf
from airontools.constructors.models.unsupervised.vae import VAE
from numpy.random import normal

tabular_data = np.concatenate(
    [
        normal(loc=0.5, scale=1, size=(100, 10)),
        normal(loc=-0.5, scale=1, size=(100, 10)),
    ]
)
model = VAE(
    input_shape=tabular_data.shape[1:],
    latent_dim=3,
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model.fit(
    tabular_data,
    epochs=10,
)
print(f"VAE evaluation: {model.evaluate(tabular_data)['loss']:.4f}")

```

### More examples

see usage examples in [aironsuit/examples](https://github.com/AtrejuArtax/aironsuit/tree/master/examples)