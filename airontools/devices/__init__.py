from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.devices.devices_torch import *
else:
    from airontools.devices.devices_tf import *
