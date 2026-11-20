"""Device-generic stream and event APIs (torch.Stream, torch.Event, torch.accelerator) on neuron.

The overlapped-output pattern vLLM's async scheduling uses (a non-blocking device-to-host copy
on a side stream that waits for the default stream, and a host wait on an event recorded after
it), written with the device-generic APIs instead of torch.cuda.
"""

import pytest
import torch

import torch_neuronx


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAcceleratorStreams:
    def test_current_stream_is_neuron(self):
        stream = torch.accelerator.current_stream()
        assert stream.device.type == "neuron"

    def test_stream_context_sets_current_stream(self):
        side = torch.Stream(device="neuron")
        default = torch.accelerator.current_stream()
        with side:
            assert torch.accelerator.current_stream() == side
        assert torch.accelerator.current_stream() == default

    def test_overlapped_output_copy(self):
        x = torch.randn(4096, 1024, device="neuron")
        y = (x * 3).sum(dim=-1)  # default-stream work producing the output

        copy_stream = torch.Stream(device="neuron")
        ready = torch.Event(device="neuron", blocking=True)
        default = torch.accelerator.current_stream()
        with copy_stream:
            copy_stream.wait_stream(default)
            y_cpu = y.to("cpu", non_blocking=True)
            ready.record()
        ready.synchronize()

        # Bit-identical to a blocking read of the same device result.
        torch.testing.assert_close(y_cpu, y.cpu(), rtol=0, atol=0)

    def test_event_orders_streams(self):
        a = torch.randn(1024, device="neuron")
        producer = torch.Stream(device="neuron")
        done = torch.Event(device="neuron")
        producer.wait_stream(torch.accelerator.current_stream())
        with producer:
            b = a + 1
            done.record()
        torch.accelerator.current_stream().wait_event(done)
        torch.testing.assert_close((b * 2).cpu(), (a.cpu() + 1) * 2)
        assert done.query()
