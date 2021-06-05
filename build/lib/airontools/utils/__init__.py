from aironsuit.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.utils.utils_tf import *
else:
    from airontools.utils.utils_torch import *
