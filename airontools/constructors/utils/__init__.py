from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.constructors.utils.utils_torch import *
else:
    from airontools.constructors.utils.utils_tf import *
