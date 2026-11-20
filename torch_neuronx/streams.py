from collections.abc import Mapping
from typing import Optional, Union

import torch

from . import _C

_NeuronStreamBase = _C._NeuronStreamBase
_NeuronEventBase = _C._NeuronEventBase


class Stream(_NeuronStreamBase):
    """Represent a logical sequence of Neuron device execution.

    Use a stream as a context manager to submit operators in its execution sequence.

    Args:
        device (torch.device or int, optional): Neuron device on which to allocate
            the stream. Defaults to ``None`` (the current device); negative uses the current device.
        priority (int, optional): Priority value encoded in the stream identifier. Defaults to
            ``0``.
        **kwargs (Any): Existing ``stream_id``, ``device_index``, and ``device_type`` fields used
            to reconstruct a stream.

    Note:
        ``TORCH_NEURONX_ENABLE_STREAMS=0`` aliases new streams to the default stream. When
        enabled, logical stream IDs select from a fixed pool containing one underlying queue.
        ``TORCH_NEURONX_ENABLE_HOST_CC`` separately controls collectives on non-default streams.
        Priority uses two signed identifier bits, is recorded as metadata rather than consumed for
        scheduling, and wraps outside ``[-2, 1]`` when read back.
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # Check if we're creating from existing stream parameters
        if "stream_id" in kwargs and "device_index" in kwargs and "device_type" in kwargs:
            return super().__new__(
                cls,
                stream_id=kwargs["stream_id"],
                device_index=kwargs["device_index"],
                device_type=kwargs["device_type"],
            )

        # Setting device manager is expensive, so we avoid it unless necessary
        if device is None:
            return super().__new__(cls, priority=priority)
        else:
            # TODO: Fix properly, right now we import here to avoid circular imports
            import torch_neuronx

            with torch_neuronx.device(device):
                return super().__new__(cls, priority=priority)

    def wait_event(self, event: "Event") -> None:
        """Make future work submitted to this stream wait for an event.

        Args:
            event (Event): Event that must complete before subsequent work can run.

        Raises:
            RuntimeError: If ``event`` is not a Neuron event.
        """
        super().wait_event(event)

    def wait_stream(self, stream: "Stream") -> None:
        """Make future work submitted to this stream wait for another stream.

        Args:
            stream (Stream): Stream whose previously submitted work must complete.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event: Optional["Event"] = None) -> "Event":
        """Record an event after work already submitted to this stream.

        Args:
            event (Event, optional): Event to record. Defaults to ``None``, which
                creates a new event.

        Returns:
            Event: Recorded event, including the supplied object when one is provided.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        """Check whether all work submitted to this stream has completed.

        Returns:
            bool: ``True`` when the stream has no outstanding work.
        """
        return super().query()

    def synchronize(self) -> None:
        """Wait for all work submitted to this stream to complete."""
        return super().synchronize()

    @property
    def device(self):
        """Return the Neuron device associated with this stream.

        Returns:
            torch.device: Neuron device containing this stream.
        """
        return torch.device("neuron", self.device_index)


class Event(_NeuronEventBase):
    """Represent a synchronization marker for Neuron device work.

    Use events to monitor completion, measure elapsed time, and order work across streams.

    Args:
        enable_timing (bool, optional): Whether the event records timing data. Defaults to
            ``False``.
        blocking (bool, optional): Stored synchronization preference, currently unused.
            Defaults to ``False``.

    Note:
        A timed event stamps its steady-clock time when the marker completes: immediately for an
        idle stream or when the queued event signal retires for a busy stream.
    """

    def __new__(cls, enable_timing=False, blocking=False):
        return super().__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
        )

    def record(self, stream: Optional["Stream"] = None) -> None:
        """Record this event after work already submitted to a stream.

        Args:
            stream (Stream, optional): Stream on which to record. Defaults to
                ``None``, which uses the current stream.
        """
        if stream is None:
            # Import here to avoid circular import
            import torch_neuronx

            stream = torch_neuronx.current_stream()
        super().record(stream)

    def wait(self, stream: Optional["Stream"] = None) -> None:
        """Make future work submitted to a stream wait for this event.

        Args:
            stream (Stream, optional): Waiting stream. Defaults to ``None``, which uses
                the current stream.

        Raises:
            RuntimeError: If ``stream`` is not a Neuron stream.
        """
        if stream is None:
            # Import here to avoid circular import
            import torch_neuronx

            stream = torch_neuronx.current_stream()
        super().wait(stream)

    def query(self) -> bool:
        """Check whether all work captured by this event has completed.

        Returns:
            bool: ``True`` when the event is unrecorded or its recorded work has completed.
        """
        return super().query()

    def elapsed_time(self, end_event: "Event") -> float:
        """Return elapsed time between two completed event markers.

        Args:
            end_event (Event): Event marking the end of the measured interval.

        Returns:
            float: Milliseconds between the events' completion timestamps.

        Raises:
            RuntimeError: If ``end_event`` is not a Neuron event, timing is disabled, either
                event is unrecorded or incomplete, or the events use different devices.

        Note:
            This method does not synchronize; call ``synchronize`` on incomplete events first.
        """
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        """Wait for all work captured by this event to complete."""
        super().synchronize()


def execute_nrt_model(
    model: int,
    inputs: Mapping[str, Union[torch.Tensor, int]],
    outputs: Mapping[str, Union[torch.Tensor, int]],
    stream: Optional[Stream] = None,
    name: str = "nrt_model",
) -> None:
    """Enqueue one execution of an externally loaded NRT model on a Neuron stream.

    The model was loaded by the caller with ``nrt_load`` in this process, on the logical
    core of the stream's device. The call returns immediately; the execution runs when the
    stream reaches it, like any other op on the stream: after the work submitted to the stream
    before it (including host-to-device writes of its inputs), and before the work submitted
    after it (compute, copies, device-to-host reads, and writes that overwrite its inputs).
    Use events and ``Stream.wait_stream`` to order it against other streams, and
    ``Stream.synchronize`` or an ``Event`` to wait for it on the host.

    Args:
        model (int): The ``nrt_model_t*`` handle, as an integer.
        inputs (Mapping[str, Tensor or int]): The model's input tensor names, each bound to a
            contiguous Neuron tensor on the stream's device or to an ``nrt_tensor_t*`` (int)
            the caller owns.
        outputs (Mapping[str, Tensor or int]): The model's output tensor names, bound the same
            way.
        stream (Stream, optional): Stream to enqueue on. Defaults to the current stream.
        name (str, optional): Op name shown in profiles and errors.

    Note:
        The model and every ``nrt_tensor_t*`` passed must stay valid until the execution
        completes (for example, until an ``Event`` recorded after it is complete). Torch
        tensors are kept alive by the stream like the operands of any other op. Execution
        errors surface the same way as for torch-neuronx's own ops.
    """
    if stream is None:
        import torch_neuronx

        stream = torch_neuronx.current_stream()
    _C._execute_nrt_model(
        model,
        list(inputs.items()),
        list(outputs.items()),
        stream.stream_id,
        stream.device_index,
        name,
    )
