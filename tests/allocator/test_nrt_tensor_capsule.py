"""_get_nrt_tensor_capsule on a fresh allocation.

The caching allocator binds a new allocation to its NRT tensor lazily, in the
allocation's kAlloc op on the stream worker. A caller that asks for the NRT
handle right after allocating (luminal_trainium's transport setup does) must get
a valid, bound handle without synchronizing first.
"""

import ctypes

import pytest
import torch

import torch_neuronx
from torch_neuronx import _C

pytestmark = pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices")

_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_get_pointer.restype = ctypes.c_void_p
_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

_nrt = ctypes.CDLL("libnrt.so.1")
_nrt.nrt_tensor_get_size.restype = ctypes.c_size_t
_nrt.nrt_tensor_get_size.argtypes = [ctypes.c_void_p]
_nrt.nrt_tensor_read.restype = ctypes.c_int
_nrt.nrt_tensor_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]


def _handle(t):
    handle = _get_pointer(_C._get_nrt_tensor_capsule(t), b"nrt_tensor_t")
    assert handle
    return handle


def _read(handle, offset, size):
    buf = (ctypes.c_uint8 * size)()
    assert _nrt.nrt_tensor_read(handle, buf, offset, size) == 0
    return bytes(buf)


@pytest.mark.parametrize("nbytes", [1, 4096, 6_442_444_800], ids=["1B", "4KiB", "6GiB"])
def test_capsule_of_fresh_allocation(nbytes):
    # Ask for the handle immediately after allocating, with nothing synchronized.
    x = torch.empty(nbytes, dtype=torch.uint8, device="neuron")
    handle = _handle(x)
    assert _nrt.nrt_tensor_get_size(handle) >= nbytes
    # The handle is the tensor's memory: what the device writes, NRT reads back.
    x.fill_(0x5A)
    assert x[-1:].cpu().item() == 0x5A  # orders the fill before the raw reads
    assert _handle(x) == handle
    for offset in sorted({0, nbytes // 2, nbytes - 1}):
        assert _read(handle, offset, 1) == b"\x5a", f"byte {offset}"


def test_capsule_after_queued_work():
    # A fresh allocation queued behind other work on the stream still gets a
    # bound handle.
    a = torch.ones(1 << 20, device="neuron")
    for _ in range(8):
        a = a * 1.0001 + 1
    y = torch.empty(1 << 20, dtype=torch.float32, device="neuron")
    handle = _handle(y)
    assert _nrt.nrt_tensor_get_size(handle) >= y.numel() * 4
    y.copy_(a)
    torch.testing.assert_close(y.cpu(), a.cpu())
