from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.interactors.interactors_torch import *
else:
    from airontools.interactors.interactors_tf import *
