from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.model_interactors.model_interactors_tf import *
else:
    from airontools.model_interactors.model_interactors_torch import *
