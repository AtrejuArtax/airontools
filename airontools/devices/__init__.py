from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.devices.devices_tf import *
else:
    from airontools.devices.devices_torch import *
