from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.model_constructors.model_constructors_tf import *
    from airontools.model_constructors.utils_tf import *
else:
    from airontools.model_constructors.model_constructors_torch import *
    from airontools.model_constructors.utils_torch import *
