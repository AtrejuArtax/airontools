from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.model_constructors.constructors_torch import *
    from airontools.model_constructors.utils_torch import *
else:
    from airontools.model_constructors.constructors_tf import *
    from airontools.model_constructors.utils_tf import *
