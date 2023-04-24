from typing import List

from tensorflow.python.client import device_lib


def get_available_gpus() -> List[str]:
    """Gets available devices.
    Returns:
        A list of device names.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]
