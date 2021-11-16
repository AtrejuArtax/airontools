from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.constructors.models.unsupervised.ae_torch import *
    from airontools.constructors.models.unsupervised.vae_torch import *
else:
    from airontools.constructors.models.unsupervised.ae_tf import *
    from airontools.constructors.models.unsupervised.vae_tf import *
