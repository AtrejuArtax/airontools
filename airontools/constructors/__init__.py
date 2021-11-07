from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.constructors.constructors_torch import *
    from airontools.constructors.utils_torch import *
else:
    from airontools.constructors.constructors_tf import *
    from airontools.constructors.utils_tf import *
