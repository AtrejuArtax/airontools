from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.visualization.tensorboard_tf import *
    from airontools.visualization.utils_tf import *
else:
    pass
