from airontools.devices import get_available_gpus
import time


def test_get_available_devices():
    time.sleep(10)
    assert isinstance(get_available_gpus(), list)
