from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.constructors.layers.layers_torch import *
else:
    from airontools.constructors.layers.layers_tf import *
    from airontools.constructors.layers.functions_tf import *
