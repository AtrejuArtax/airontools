from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.visualization.tensorboard_torch import *
else:
    from airontools.visualization.tensorboard_tf import *
