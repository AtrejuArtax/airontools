# AIronTools

AIronTools (Beta) is a Python library that provides the user with higher level deep learning tools built to work with 
[tensorflow](https://github.com/tensorflow/tensorflow) as a backend.

Key features:

1. Model constructor that allows multiple models to be optimized in parallel across multiple GPUs. 
2. Block constructor to build customised blocks/models.
3. Layer constructor to build customised layers.
4. Preprocessing utils.
5. On the fly non-topological hyper-parameter optimization. For now only the dropout regularization is compatible with this feature, in the future others such as l1 and l2 regularization will be compatible too.
6. Save latent representations for visualization purposes.
   
### Installation

`pip install airontools`

### Custom Keras subclass to build a variational autoencoder (VAE) with airontools and compatible with aironsuit

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
print("VAE evaluation:", float(model.evaluate(tabular_data)["loss"]))

```

### More examples

see usage examples in [aironsuit/examples](https://github.com/AtrejuArtax/aironsuit/tree/master/examples)