# tests/test_device.py
import torch
from microscale.device import DeviceInfo, Runtime, get_device, get_torch_device, device_summary


def test_get_device_returns_device_info():
    info = get_device()
    assert isinstance(info, DeviceInfo)
    assert isinstance(info.runtime, Runtime)
    assert isinstance(info.name, str)
    assert isinstance(info.description, str)
    assert info.name in ("cuda", "mps", "cpu", "mlx")


def test_get_torch_device_returns_torch_device():
    dev = get_torch_device()
    assert isinstance(dev, torch.device)


def test_tensor_creation_on_detected_device():
    dev = get_torch_device()
    x = torch.randn(4, 4, device=dev)
    assert x.device.type == dev.type


def test_device_summary_returns_string():
    summary = device_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
