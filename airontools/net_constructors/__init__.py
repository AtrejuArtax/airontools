from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.net_constructors.net_constructors_tf import *
else:
    pass
