"""
The device-generic torch.accelerator memory APIs reach the Neuron caching allocator.

The allocator is registered as a c10::DeviceAllocator, so torch.accelerator.empty_cache(),
memory_stats(), reset_*_stats() and get_memory_info() work on the neuron device and agree with
torch_neuronx's own memory APIs.
"""

import pytest
import torch

import torch_neuronx


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAcceleratorMemory:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        torch_neuronx.empty_cache()
        torch_neuronx.reset_peak_memory_stats()
        torch_neuronx.reset_accumulated_memory_stats()
        yield
        torch_neuronx.empty_cache()

    def test_accelerator_is_neuron(self):
        assert torch.accelerator.current_accelerator().type == "neuron"

    def test_empty_cache_frees_reserved_bytes(self):
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        del x
        torch_neuronx.synchronize()
        assert torch.accelerator.memory_reserved() > 0

        torch.accelerator.empty_cache()

        assert torch.accelerator.memory_reserved() == 0
        assert torch_neuronx.memory_reserved() == 0
        assert torch.accelerator.memory_allocated() == 0

    def test_memory_stats_match_torch_neuronx(self):
        x = torch.randn(4096, device="neuron")
        torch_neuronx.synchronize()
        acc = torch.accelerator.memory_stats()
        own = torch_neuronx.memory_stats()
        for cat in ("allocated_bytes", "reserved_bytes", "active_bytes"):
            for metric in ("current", "peak", "allocated", "freed"):
                key = f"{cat}.all.{metric}"
                assert acc[key] == own[key], key
        assert acc["active_bytes.all.current"] >= x.numel() * x.element_size()
        assert torch.accelerator.memory_allocated() == torch_neuronx.memory_allocated()
        del x

    def test_reset_peak_memory_stats(self):
        x = torch.randn(1 << 16, device="neuron")
        torch_neuronx.synchronize()
        peak = torch.accelerator.max_memory_allocated()
        del x
        torch_neuronx.synchronize()
        assert torch.accelerator.max_memory_allocated() == peak
        assert torch.accelerator.memory_allocated() < peak

        torch.accelerator.reset_peak_memory_stats()

        assert torch.accelerator.max_memory_allocated() == torch.accelerator.memory_allocated()

    def test_reset_accumulated_memory_stats(self):
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        del x
        torch_neuronx.synchronize()
        assert torch.accelerator.memory_stats()["allocated_bytes.all.allocated"] > 0

        torch.accelerator.reset_accumulated_memory_stats()

        stats = torch.accelerator.memory_stats()
        assert stats["allocated_bytes.all.allocated"] == 0
        assert stats["allocated_bytes.all.freed"] == 0

    def test_get_memory_info(self):
        free, total = torch.accelerator.get_memory_info()
        assert total == torch_neuronx.get_device_properties(0).total_memory
        assert 0 < free <= total
