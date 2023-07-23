import os
import shutil


def path_management(path: str, modes=None) -> None:
    """Path preparation.

    Parameters:
        path (str): Path to manage.
        modes (list): Mode per path, remove as 'rm' and make as 'make'.
    """
    available_modes = ["rm", "make"]
    if not modes:
        modes = ["make"]
    for mode in modes:
        assert mode in available_modes
        if os.path.isdir(path) and mode == "rm":
            shutil.rmtree(path)
        elif mode == "make":
            os.makedirs(path, exist_ok=True)
