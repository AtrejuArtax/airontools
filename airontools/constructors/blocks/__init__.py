from airontools.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'pytorch':
    from airontools.constructors.blocks.blocks_torch import *
else:
    from airontools.constructors.blocks.blocks_tf import *
