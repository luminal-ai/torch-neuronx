"""E2E tests for ``torch_neuronx.execute_nrt_model``.

The tests load a NEFF themselves with ``nrt_load`` (as an external library would) and enqueue
its executions on Neuron streams. They check that the execution is ordered like any other
stream op: after earlier host-to-device writes of its inputs, before later device-to-host
reads and eager ops on its outputs, and before later writes that overwrite its inputs.
"""

import ctypes
import os
import tempfile

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch

import torch_neuronx
import torch_neuronx._C as _C

SHAPE = (128, 512)


@nki.jit
def _add(input0, input1):
    sa = nl.ndarray(input0.shape, dtype=input0.dtype, buffer=nl.sbuf)
    sb = nl.ndarray(input1.shape, dtype=input1.dtype, buffer=nl.sbuf)
    sc = nl.ndarray(input0.shape, dtype=input0.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sa, src=input0)
    nisa.dma_copy(dst=sb, src=input1)
    nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.add)
    output0 = nl.ndarray(input0.shape, dtype=input0.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output0, src=sc)
    return output0


def _compile_add_neff(shape):
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    from nki.framework.kernel import Kernel

    with tempfile.TemporaryDirectory() as td:
        a = np.empty(shape, dtype=np.float32)
        opts = CompileOptions(
            target=_C._get_platform_target(),
            verbose=False,
            artifacts_dir=td,
            output_path=os.path.join(td, "out.neff"),
        )
        nir = compile_to_bir(
            Kernel(_add.func if hasattr(_add, "func") else _add),
            frontend=ParserFrontend(),
            inputs={"input0": a, "input1": a},
            compile_opts=opts,
            output_names=["output0"],
        )
        compile_bir_to_neff(
            opts,
            nir,
            input_arrays=[],
            argument_names=[s.name for s in nir.descriptor.input_specs],
            output_arg_names=[s.name for s in nir.descriptor.output_specs],
        )
        with open(os.path.join(td, "out.neff"), "rb") as f:
            return f.read()


def _libnrt():
    lib = ctypes.CDLL("libnrt.so.1")
    lib.nrt_load.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.nrt_load.restype = ctypes.c_int
    lib.nrt_unload.argtypes = [ctypes.c_void_p]
    lib.nrt_unload.restype = ctypes.c_int
    return lib


@pytest.fixture(scope="module")
def add_model():
    """An add model loaded outside torch-neuronx, on device 0's logical core."""
    torch.zeros(1, device="neuron")  # initialize the runtime
    neff = _compile_add_neff(SHAPE)
    lib = _libnrt()
    model = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(neff, len(neff))
    status = lib.nrt_load(buf, len(neff), _C._get_vnc_id(0), 1, ctypes.byref(model))
    assert status == 0, f"nrt_load failed: {status}"
    yield model.value
    torch_neuronx.synchronize()
    lib.nrt_unload(model)


def _raw_nrt_tensor(t):
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return get(_C._get_nrt_tensor_capsule(t), b"nrt_tensor_t")


def test_result_visible_to_later_reads(add_model):
    a_cpu, b_cpu = torch.randn(SHAPE), torch.randn(SHAPE)
    a, b = a_cpu.to("neuron"), b_cpu.to("neuron")
    out = torch.empty(SHAPE, device="neuron")

    torch_neuronx.execute_nrt_model(add_model, {"input0": a, "input1": b}, {"output0": out})

    # No synchronize: the D2H read is ordered after the execution by the stream.
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu, rtol=0, atol=0)


def test_eager_ops_consume_output(add_model):
    a_cpu, b_cpu = torch.randn(SHAPE), torch.randn(SHAPE)
    a, b = a_cpu.to("neuron"), b_cpu.to("neuron")
    out = torch.empty(SHAPE, device="neuron")

    torch_neuronx.execute_nrt_model(add_model, {"input0": a, "input1": b}, {"output0": out})
    rows = torch.tensor([3, 0, 77], device="neuron")
    picked = out[rows].sum(dim=-1)

    torch.testing.assert_close(picked.cpu(), (a_cpu + b_cpu)[[3, 0, 77]].sum(dim=-1))


def test_overwriting_inputs_waits_for_execution(add_model):
    a_cpu, b_cpu = torch.randn(SHAPE), torch.randn(SHAPE)
    a, b = a_cpu.to("neuron"), b_cpu.to("neuron")
    out = torch.empty(SHAPE, device="neuron")

    torch_neuronx.execute_nrt_model(add_model, {"input0": a, "input1": b}, {"output0": out})
    # The next step's inputs: written into the same buffers right after the enqueue.
    a.copy_(torch.zeros(SHAPE), non_blocking=True)
    b.copy_(torch.zeros(SHAPE), non_blocking=True)

    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu, rtol=0, atol=0)
    assert torch.count_nonzero(a.cpu()) == 0


def test_chained_executions_and_views(add_model):
    a_cpu = torch.randn(2 * SHAPE[0], SHAPE[1])
    base = a_cpu.to("neuron")
    lo, hi = base[: SHAPE[0]], base[SHAPE[0] :]  # contiguous views at an offset
    out = torch.empty(SHAPE, device="neuron")
    out2 = torch.empty(SHAPE, device="neuron")

    torch_neuronx.execute_nrt_model(add_model, {"input0": lo, "input1": hi}, {"output0": out})
    torch_neuronx.execute_nrt_model(add_model, {"input0": out, "input1": hi}, {"output0": out2})

    expected = a_cpu[: SHAPE[0]] + a_cpu[SHAPE[0] :] + a_cpu[SHAPE[0] :]
    torch.testing.assert_close(out2.cpu(), expected)


def test_raw_nrt_tensor_bindings(add_model):
    a_cpu, b_cpu = torch.randn(SHAPE), torch.randn(SHAPE)
    a, b = a_cpu.to("neuron"), b_cpu.to("neuron")
    out = torch.empty(SHAPE, device="neuron")
    torch_neuronx.synchronize()  # bind the output's memory before taking its handle
    raw_out = _raw_nrt_tensor(out)

    torch_neuronx.execute_nrt_model(add_model, {"input0": a, "input1": b}, {"output0": raw_out})

    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu, rtol=0, atol=0)


def test_side_stream_and_events(add_model):
    a_cpu, b_cpu = torch.randn(SHAPE), torch.randn(SHAPE)
    a, b = a_cpu.to("neuron"), b_cpu.to("neuron")
    out = torch.empty(SHAPE, device="neuron")
    side = torch_neuronx.Stream()
    side.wait_stream(torch_neuronx.current_stream())

    torch_neuronx.execute_nrt_model(
        add_model, {"input0": a, "input1": b}, {"output0": out}, stream=side
    )
    done = side.record_event()
    torch_neuronx.current_stream().wait_event(done)

    torch.testing.assert_close((out * 2).cpu(), (a_cpu + b_cpu) * 2)


def test_rejects_tensor_on_wrong_device_type(add_model):
    a = torch.randn(SHAPE)
    out = torch.empty(SHAPE, device="neuron")
    with pytest.raises(RuntimeError, match="expected neuron"):
        torch_neuronx.execute_nrt_model(add_model, {"input0": a, "input1": a}, {"output0": out})
