from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.net_interactors.net_interactors_tf import *
else:
    from airontools.net_interactors.net_interactors_torch import *
