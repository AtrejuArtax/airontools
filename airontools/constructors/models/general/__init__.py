from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.constructors.models.general.general_torch import *
else:
    from airontools.constructors.models.general.general_tf import *
