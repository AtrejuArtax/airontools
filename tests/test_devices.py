from airontools.devices import get_available_gpus


def test_get_available_devices():
    assert isinstance(get_available_gpus(), list)
