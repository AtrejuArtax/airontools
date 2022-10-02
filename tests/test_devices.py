from airontools.devices import get_available_gpus


def test_get_available_devices():
    available_gpus = get_available_gpus()
    assert isinstance(available_gpus, list)
    assert all(isinstance(element, str) for element in available_gpus)
