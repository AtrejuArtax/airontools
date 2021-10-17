from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from airontools.visualization.tensorboard_tf import *
    from airontools.visualization.utils_tf import *
else:
    from airontools.visualization.tensorboard_torch import *
    from airontools.visualization.utils_torch import *
