"""Fills and filled factories of tensors with 2^31 or more elements, at exact size.

The Neuron runtime keeps every NEFF tensor dimension in a uint32, so an eager fill
whose tensor has a larger dimension must not become one NEFF over that shape
(the runtime refuses to load it: "NMGR:extract_tensor_info Invalid shape
dimension"). These tests fill such tensors and check every element on device
(min and max over the whole tensor equal the fill value) plus host reads at the
boundaries a split could get wrong. vLLM's KV cache allocation is
torch.zeros(6442444800, dtype=int8) on the device.
"""

import gc

import pytest
import torch

import torch_neuronx

pytestmark = pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices")

DEV = "neuron"
# Row length of the 2-D views the checks reduce over; any value whose views have
# dimensions well inside the runtime limit works.
CHECK_ROW = 1 << 20
# Rows per checked chunk (1 GiB of words), bounding the check's temporaries.
CHECK_ROWS = 256


def _dense_flat(x):
    assert x.is_contiguous()
    return x.view(-1)


def _element_bytes(value, dtype):
    return torch.tensor([value], dtype=dtype).view(torch.uint8).tolist()


def _check_words(flat, pattern):
    """Every byte of the dense 1-D device tensor `flat` follows the repeating byte
    `pattern` (whose length divides 4), checked on device.

    The bytes are compared as int32 words: a uint8/int8 reduction widens its input to
    an int32 copy, 4x the tensor. Neuron reduces int32 through float32, which is not
    exact above 2^24, so each chunk is XORed with the expected word and its min and
    max must be 0 (a nonzero int32 stays nonzero in float32). The last nbytes % 4
    bytes are read on the host.
    """
    raw = flat.view(torch.uint8)
    nbytes = raw.numel()
    nwords = nbytes // 4
    word = int.from_bytes(bytes(pattern * (4 // len(pattern))), "little", signed=True)
    words = raw[: nwords * 4].view(torch.int32)
    rows, tail = divmod(nwords, CHECK_ROW)
    chunk = CHECK_ROWS * CHECK_ROW
    end = rows * CHECK_ROW
    parts = [words[r : min(r + chunk, end)].view(-1, CHECK_ROW) for r in range(0, end, chunk)]
    if tail:
        parts.append(words[rows * CHECK_ROW :])
    for part in parts:
        diff = torch.bitwise_xor(part, word)
        hi = diff.amax().cpu().item()
        lo = diff.amin().cpu().item()
        assert hi == 0 and lo == 0, f"words differ from {word:#x} (xor min {lo} max {hi})"
    rest = raw[nwords * 4 :].cpu().tolist()
    expect = [pattern[(nwords * 4 + i) % len(pattern)] for i in range(len(rest))]
    assert rest == expect, f"last bytes {rest} != {expect}"


def _check_values(flat, value):
    """Every element of the dense 1-D device tensor `flat` equals `value`."""
    _check_words(flat, _element_bytes(value, flat.dtype))


def _boundary_indices(n):
    """Indices around 2^30 multiples (the split's row length), 2^31, 2^32 and both ends."""
    marks = {0, n - 1}
    for edge in (1 << 30, 1 << 31, 1 << 32, (n // (1 << 30)) * (1 << 30)):
        marks.update({edge - 1, edge, edge + 1})
    return sorted(i for i in marks if 0 <= i < n)


def _check_points(flat, value):
    for i in _boundary_indices(flat.numel()):
        got = flat[i : i + 1].cpu().item()
        assert got == value, f"element {i} is {got}, expected {value}"


def _check_all(x, value):
    flat = _dense_flat(x)
    _check_points(flat, value)
    _check_values(flat, value)


@pytest.fixture(autouse=True)
def _free_device_memory():
    yield
    gc.collect()


SIZES = [
    pytest.param(1, id="1"),
    pytest.param((1 << 31) - 1, id="2^31-1"),
    pytest.param(1 << 31, id="2^31"),
    pytest.param((1 << 32) + 7, id="2^32+7"),
]


@pytest.mark.parametrize("n", SIZES)
def test_fill_then_zero_int8(n):
    x = torch.empty(n, dtype=torch.int8, device=DEV)
    assert x.fill_(5) is x
    _check_all(x, 5)
    assert x.zero_() is x
    _check_all(x, 0)
    assert x.shape == (n,) and x.dtype == torch.int8


@pytest.mark.parametrize("n", SIZES)
def test_zeros_int8(n):
    # Make the allocator hand back memory holding a nonzero pattern first, so a
    # missing write shows up instead of reading leftover zeros.
    junk = torch.empty(n, dtype=torch.int8, device=DEV).fill_(-3)
    _check_points(_dense_flat(junk), -3)
    del junk
    x = torch.zeros(n, dtype=torch.int8, device=DEV)
    assert x.shape == (n,) and x.dtype == torch.int8 and x.device.type == "neuron"
    _check_all(x, 0)


def test_vllm_kv_cache_zeros_uint8():
    """The 6 GiB KV cache buffer: 6,442,444,800 bytes, every byte zero."""
    n = 6_442_444_800
    junk = torch.empty(n, dtype=torch.uint8, device=DEV).fill_(0xA5)
    _check_all(junk, 0xA5)
    del junk
    x = torch.zeros(n, dtype=torch.uint8, device=DEV)
    assert x.shape == (n,) and x.dtype == torch.uint8
    _check_all(x, 0)


def test_ones_full_tensor_fill_large():
    n = (1 << 32) + 7
    x = torch.ones(n, dtype=torch.int8, device=DEV)
    _check_all(x, 1)
    del x
    x = torch.full((n,), 3, dtype=torch.uint8, device=DEV)
    _check_all(x, 3)
    x.fill_(torch.tensor(9, dtype=torch.uint8, device=DEV))
    _check_all(x, 9)
    del x
    x = torch.ones_like(torch.empty(n, dtype=torch.int8, device=DEV))
    _check_all(x, 1)


def test_zeros_out_large():
    n = (1 << 32) + 7
    out = torch.empty(n, dtype=torch.int8, device=DEV).fill_(4)
    res = torch.zeros((n,), out=out)
    assert res is out
    _check_all(out, 0)


def test_fill_bfloat16_large():
    n = (1 << 32) + 7
    x = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    x.fill_(1.5)
    _check_all(x, 1.5)
    x.zero_()
    _check_all(x, 0.0)


def test_fill_multidim_dense_large():
    # A dense 2-D tensor whose last dimension is over the limit, filled through a
    # transposed (dense, non-contiguous) view.
    n = (1 << 32) + 7
    x = torch.zeros(2, n, dtype=torch.int8, device=DEV)
    _check_all(x, 0)
    x.t().fill_(6)
    _check_all(x, 6)


def test_fill_strided_large_leaves_gaps():
    # A strided view with a dimension over the limit: only its elements change.
    n = (1 << 32) + 7
    base = torch.empty(n, 2, dtype=torch.int8, device=DEV).fill_(1)
    col = base[:, 0]
    assert not col.is_contiguous()
    col.fill_(8)
    flat = _dense_flat(base)
    for i in _boundary_indices(n):
        assert flat[2 * i : 2 * i + 2].cpu().tolist() == [8, 1], f"row {i}"
    # Every even byte (column 0) is 8 and every odd byte (column 1) still 1.
    _check_words(flat, [8, 1])


@pytest.mark.parametrize("shape", [(), (1,), (3, 4), (2, 3, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32, torch.int8])
def test_small_fills_unchanged(shape, dtype):
    x = torch.empty(shape, dtype=dtype, device=DEV)
    x.fill_(3)
    assert torch.equal(x.cpu(), torch.full(shape, 3, dtype=dtype))
    x.zero_()
    assert torch.equal(x.cpu(), torch.zeros(shape, dtype=dtype))
    assert torch.equal(torch.zeros(shape, dtype=dtype, device=DEV).cpu(), torch.zeros(shape, dtype=dtype))
    assert torch.equal(torch.ones(shape, dtype=dtype, device=DEV).cpu(), torch.ones(shape, dtype=dtype))
    assert torch.equal(
        torch.full(shape, 7, dtype=dtype, device=DEV).cpu(), torch.full(shape, 7, dtype=dtype)
    )
