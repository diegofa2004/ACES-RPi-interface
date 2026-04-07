"""Helpers for bridging raw FPGA audio to Raspberry Pi."""

from .compararEvento import compararEvento
from .fpga_audio_adapter import AudioCaptureConfig, FPGAAudioReceiver

__all__ = [
    "compararEvento",
    "AudioCaptureConfig",
    "FPGAAudioReceiver",
]
