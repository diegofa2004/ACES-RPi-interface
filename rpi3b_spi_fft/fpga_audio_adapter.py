import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from .i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_BACKEND,
        DEFAULT_CAPTURE_RATE_HZ,
        DEFAULT_CHANNEL_COUNT,
        DEFAULT_READ_FRAMES,
        build_alsa_capture_command,
        build_arecord_command,
        resolve_audio_device,
        resolve_capture_backend,
    )
    from .spectral_features import build_dct_matrix, build_mel_filter
except ImportError:
    from i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_BACKEND,
        DEFAULT_CAPTURE_RATE_HZ,
        DEFAULT_CHANNEL_COUNT,
        DEFAULT_READ_FRAMES,
        build_alsa_capture_command,
        build_arecord_command,
        resolve_audio_device,
        resolve_capture_backend,
    )
    from spectral_features import build_dct_matrix, build_mel_filter


@dataclass
class AudioCaptureConfig:
    device: str = AUTO_AUDIO_DEVICE
    sample_rate: int = DEFAULT_CAPTURE_RATE_HZ
    frame_length: int = 512
    useful_bins: int = 256
    capture_backend: str = DEFAULT_CAPTURE_BACKEND
    capture_binary: Optional[str] = None
    read_frames: int = DEFAULT_READ_FRAMES
    channels: int = DEFAULT_CHANNEL_COUNT
    sample_shift_bits: int = 8
    mono_channel: str = "auto"
    auto_channel_ratio: float = 4.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_length <= 0:
            raise ValueError("frame_length must be positive")
        if self.read_frames <= 0:
            raise ValueError("read_frames must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if not 0 <= self.sample_shift_bits <= 16:
            raise ValueError("sample_shift_bits must be between 0 and 16")
        max_useful_bins = (self.frame_length // 2) + 1
        if not 2 <= self.useful_bins <= max_useful_bins:
            raise ValueError("Expected 2 <= useful_bins <= (frame_length // 2) + 1")
        if self.mono_channel not in {"auto", "left", "right", "average"}:
            raise ValueError("mono_channel must be one of: auto, left, right, average")
        if self.auto_channel_ratio <= 1.0:
            raise ValueError("auto_channel_ratio must be greater than 1")


class FPGAAudioReceiver:
    def __init__(self, cfg: AudioCaptureConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._byte_buffer = bytearray()
        self._frame_bytes = self.cfg.frame_length * self.cfg.channels * 4
        self._read_chunk_bytes = self.cfg.read_frames * self.cfg.channels * 4
        self._window = np.hanning(self.cfg.frame_length).astype(np.float32)
        self._useful_bins = int(self.cfg.useful_bins)
        mel_filter = build_mel_filter(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.frame_length,
            n_mels=32,
        )
        self.mel_filter = np.asarray(mel_filter[:, : self._useful_bins], dtype=np.float32)
        self._dct_matrix = build_dct_matrix(input_size=32, output_size=13)

    def start(self) -> None:
        resolved_device = resolve_audio_device(self.cfg.device)
        backend = resolve_capture_backend(self.cfg.capture_backend, self.cfg.capture_binary)
        self.cfg.device = resolved_device
        self.cfg.capture_backend = backend
        self._byte_buffer.clear()

        if backend == "alsa-c":
            command = build_alsa_capture_command(
                resolved_device,
                sample_rate=self.cfg.sample_rate,
                read_frames=self.cfg.read_frames,
                capture_binary=self.cfg.capture_binary,
            )
        else:
            command = build_arecord_command(
                resolved_device,
                sample_rate=self.cfg.sample_rate,
            )

        try:
            self._proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Unable to start audio capture backend {backend!r}. "
                "Check whether the requested binary is installed."
            ) from exc

        if self._proc.stdout is None:
            raise RuntimeError("Audio capture backend did not expose stdout")

        print(
            "Starting ALSA/I2S capture:",
            resolved_device,
            f"backend={backend}",
            f"rate={self.cfg.sample_rate}",
            flush=True,
        )

    def stop(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
            self._proc = None
        self._byte_buffer.clear()

    def _fill_buffer(self, min_bytes: int) -> bool:
        if self._proc is None or self._proc.stdout is None:
            return False

        while len(self._byte_buffer) < min_bytes:
            chunk = self._proc.stdout.read(max(self._read_chunk_bytes, min_bytes - len(self._byte_buffer)))
            if not chunk:
                break
            self._byte_buffer.extend(chunk)

        return len(self._byte_buffer) >= min_bytes

    def _select_mono_channel(self, pcm_frames: np.ndarray) -> np.ndarray:
        if pcm_frames.ndim != 2 or pcm_frames.shape[0] == 0:
            return np.empty(0, dtype=np.int32)
        if pcm_frames.shape[1] == 1:
            return pcm_frames[:, 0]

        left = pcm_frames[:, 0]
        right = pcm_frames[:, 1]

        if self.cfg.mono_channel == "left":
            return left
        if self.cfg.mono_channel == "right":
            return right
        if self.cfg.mono_channel == "average":
            return ((left.astype(np.int64) + right.astype(np.int64)) // 2).astype(np.int32)

        left_energy = float(np.mean(np.abs(left.astype(np.float64)), dtype=np.float64)) + 1.0
        right_energy = float(np.mean(np.abs(right.astype(np.float64)), dtype=np.float64)) + 1.0

        if left_energy >= (right_energy * self.cfg.auto_channel_ratio):
            return left
        if right_energy >= (left_energy * self.cfg.auto_channel_ratio):
            return right

        return ((left.astype(np.int64) + right.astype(np.int64)) // 2).astype(np.int32)

    def read_pcm_frame(self) -> Optional[np.ndarray]:
        if not self._fill_buffer(self._frame_bytes):
            return None

        raw = bytes(self._byte_buffer[: self._frame_bytes])
        del self._byte_buffer[: self._frame_bytes]

        pcm_frames = np.frombuffer(raw, dtype="<i4").reshape(-1, self.cfg.channels)
        mono = self._select_mono_channel(pcm_frames)
        shifted = np.right_shift(mono, self.cfg.sample_shift_bits)
        return shifted.astype(np.float32, copy=False)

    def read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pcm_frame = self.read_pcm_frame()
        if pcm_frame is None:
            return None

        windowed = pcm_frame * self._window
        fft_full = np.abs(np.fft.rfft(windowed))
        fft_useful = np.asarray(fft_full[: self._useful_bins], dtype=np.float32)

        mel = self.mel_filter @ fft_useful
        mel = np.log(mel + 1e-9)
        mfcc = self._dct_matrix @ mel

        return fft_useful, np.asarray(mfcc, dtype=np.float32)
