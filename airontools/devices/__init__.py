from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.devices.utils_tf import *
else:
    from airontools.devices.utils_torch import *
