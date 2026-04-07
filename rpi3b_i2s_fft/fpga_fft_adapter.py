import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from .i2s_stream import (
        AUTO_AUDIO_DEVICE,
        BYTES_PER_STEREO_FRAME,
        DEFAULT_CAPTURE_RATE_HZ,
        DEFAULT_FFT_PACKET_INDEX_BASE,
        DEFAULT_PACKET_INDEX_BITS,
        DEFAULT_PACKET_INDEX_SHIFT,
        DEFAULT_PAYLOAD_BITS,
        DEFAULT_TAG_BFPEXP,
        DEFAULT_TAG_FFT,
        DEFAULT_TAG_IDLE,
        DEFAULT_TAG_MASK,
        DEFAULT_TAG_SHIFT,
        build_capture_cmd,
        resolve_audio_device,
        start_capture_process,
        stop_process,
    )
    from .spectral_features import build_dct_matrix, build_mel_filter
except ImportError:
    from i2s_stream import (
        AUTO_AUDIO_DEVICE,
        BYTES_PER_STEREO_FRAME,
        DEFAULT_CAPTURE_RATE_HZ,
        DEFAULT_FFT_PACKET_INDEX_BASE,
        DEFAULT_PACKET_INDEX_BITS,
        DEFAULT_PACKET_INDEX_SHIFT,
        DEFAULT_PAYLOAD_BITS,
        DEFAULT_TAG_BFPEXP,
        DEFAULT_TAG_FFT,
        DEFAULT_TAG_IDLE,
        DEFAULT_TAG_MASK,
        DEFAULT_TAG_SHIFT,
        build_capture_cmd,
        resolve_audio_device,
        start_capture_process,
        stop_process,
    )
    from spectral_features import build_dct_matrix, build_mel_filter


CHANNEL_MODE_AUTO = "auto"
CHANNEL_MODE_LEFT = "left"
CHANNEL_MODE_RIGHT = "right"
CHANNEL_MODE_AVERAGE = "average"
DEFAULT_BFPEXP_HOLD_PAIRS = 1
DEFAULT_TAG_LOSS_TOLERANCE_PAIRS = 0


def decode_stereo_frames(raw: bytes) -> np.ndarray:
    if not raw:
        return np.empty((0, 2), dtype=np.int32)
    valid_size = len(raw) - (len(raw) % BYTES_PER_STEREO_FRAME)
    if valid_size <= 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.frombuffer(raw[:valid_size], dtype=np.int32).reshape(-1, 2)


def select_mono_channel(
    stereo: np.ndarray,
    channel_mode: str = CHANNEL_MODE_AUTO,
) -> tuple[np.ndarray, str]:
    stereo_i32 = np.asarray(stereo, dtype=np.int32)
    if stereo_i32.size == 0:
        return np.empty(0, dtype=np.int32), CHANNEL_MODE_LEFT
    if stereo_i32.ndim != 2 or stereo_i32.shape[1] != 2:
        stereo_i32 = stereo_i32.reshape(-1, 2)

    normalized_mode = (channel_mode or CHANNEL_MODE_AUTO).strip().lower()
    left = stereo_i32[:, 0]
    right = stereo_i32[:, 1]

    if normalized_mode == CHANNEL_MODE_LEFT:
        return left, CHANNEL_MODE_LEFT
    if normalized_mode == CHANNEL_MODE_RIGHT:
        return right, CHANNEL_MODE_RIGHT
    if normalized_mode == CHANNEL_MODE_AVERAGE:
        averaged = ((left.astype(np.int64) + right.astype(np.int64)) // 2).astype(np.int32)
        return averaged, CHANNEL_MODE_AVERAGE

    left_energy = float(np.mean(np.abs(left.astype(np.int64)), dtype=np.float64))
    right_energy = float(np.mean(np.abs(right.astype(np.int64)), dtype=np.float64))
    if right_energy > left_energy:
        return right, CHANNEL_MODE_RIGHT
    return left, CHANNEL_MODE_LEFT


def prepare_audio_window(
    stereo: np.ndarray,
    *,
    channel_mode: str = CHANNEL_MODE_AUTO,
    sample_shift_bits: int = 0,
    remove_dc: bool = True,
) -> tuple[np.ndarray, str]:
    mono, used_mode = select_mono_channel(stereo, channel_mode=channel_mode)
    if sample_shift_bits:
        mono = np.right_shift(mono, sample_shift_bits)

    frame = mono.astype(np.float32, copy=False)
    if remove_dc and frame.size:
        frame = frame - np.mean(frame, dtype=np.float32)
    return frame, used_mode


def compute_fft_magnitude(
    mono_frame: np.ndarray,
    *,
    frame_bins: int,
    useful_bins: int,
    window: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame_f32 = np.asarray(mono_frame, dtype=np.float32)
    if window is None:
        window_f32 = np.hanning(frame_bins).astype(np.float32)
    else:
        window_f32 = np.asarray(window, dtype=np.float32)
    fft_full = np.abs(np.fft.rfft(frame_f32 * window_f32, n=frame_bins)).astype(np.float32)
    fft_useful = fft_full[:useful_bins]
    return fft_full, fft_useful


@dataclass
class FFTAdapterConfig:
    device: str = AUTO_AUDIO_DEVICE
    sample_rate: int = DEFAULT_CAPTURE_RATE_HZ
    frame_bins: int = 512
    useful_bins: int = 256
    capture_backend: str = "auto"
    capture_binary: Optional[str] = None
    channel_mode: str = CHANNEL_MODE_AUTO
    sample_shift_bits: int = 0
    remove_dc: bool = True
    gpio_chip: str = "/dev/gpiochip0"
    bfpexp_flag_line: Optional[int] = None
    done_line: Optional[int] = None
    flag_active_high: bool = True
    done_pulse_seconds: float = 0.0005
    handshake_timeout_seconds: float = 1.0
    wait_for_flag_falling_edge: bool = True
    use_i2s_tags: bool = False
    packet_index_shift: int = DEFAULT_PACKET_INDEX_SHIFT
    packet_index_bits: int = DEFAULT_PACKET_INDEX_BITS
    fft_packet_index_base: int = DEFAULT_FFT_PACKET_INDEX_BASE
    tag_shift: int = DEFAULT_TAG_SHIFT
    tag_mask: int = DEFAULT_TAG_MASK
    payload_bits: int = DEFAULT_PAYLOAD_BITS
    tag_idle: int = DEFAULT_TAG_IDLE
    tag_bfpexp: int = DEFAULT_TAG_BFPEXP
    tag_fft: int = DEFAULT_TAG_FFT
    apply_bfpexp: bool = False
    require_bfpexp_before_fft: bool = False
    bfpexp_pairs_required: int = DEFAULT_BFPEXP_HOLD_PAIRS
    loss_tolerance_pairs: int = DEFAULT_TAG_LOSS_TOLERANCE_PAIRS

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_bins < 2:
            raise ValueError("frame_bins must be at least 2")
        max_useful_bins = 1 + (self.frame_bins // 2)
        if not 2 <= self.useful_bins <= max_useful_bins:
            raise ValueError(f"Expected 2 <= useful_bins <= {max_useful_bins}")
        if self.sample_shift_bits < 0:
            raise ValueError("sample_shift_bits must be non-negative")
        normalized_channel_mode = (self.channel_mode or CHANNEL_MODE_AUTO).strip().lower()
        if normalized_channel_mode not in (
            CHANNEL_MODE_AUTO,
            CHANNEL_MODE_LEFT,
            CHANNEL_MODE_RIGHT,
            CHANNEL_MODE_AVERAGE,
        ):
            raise ValueError("channel_mode must be one of auto, left, right, average")
        self.channel_mode = normalized_channel_mode


class FPGAFFTReceiver:
    def __init__(self, cfg: FFTAdapterConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.cfg.frame_bins * BYTES_PER_STEREO_FRAME
        self._poll_frames = max(256, self.cfg.frame_bins)
        self._poll_bytes = self._poll_frames * BYTES_PER_STEREO_FRAME
        self._byte_buffer = bytearray()
        self._window = np.hanning(self.cfg.frame_bins).astype(np.float32)
        self._mel_filter = build_mel_filter(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.frame_bins,
            n_mels=32,
        )
        self._dct_matrix = build_dct_matrix(input_size=32, output_size=13)
        self.last_channel_used = CHANNEL_MODE_LEFT
        self.last_audio_frame = np.zeros(self.cfg.frame_bins, dtype=np.float32)
        self.last_frame_bfpexp = 0
        self.last_frame_had_explicit_bfpexp = False
        self.last_frame_missing_bins: tuple[int, ...] = tuple()

    def _fill_buffer(self, min_bytes: int) -> bool:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Receiver not started")

        while len(self._byte_buffer) < min_bytes:
            chunk = self._proc.stdout.read(max(self._poll_bytes, min_bytes - len(self._byte_buffer)))
            if not chunk:
                break
            self._byte_buffer.extend(chunk)

        return len(self._byte_buffer) >= min_bytes

    def _pop_stereo_frames(self, frame_count: int, *, exact: bool) -> Optional[np.ndarray]:
        if frame_count <= 0:
            return np.empty((0, 2), dtype=np.int32)

        needed = frame_count * BYTES_PER_STEREO_FRAME
        if exact:
            if not self._fill_buffer(needed):
                return None
            raw = bytes(self._byte_buffer[:needed])
            del self._byte_buffer[:needed]
        else:
            if not self._fill_buffer(BYTES_PER_STEREO_FRAME):
                return None
            available = min(len(self._byte_buffer), needed)
            available -= available % BYTES_PER_STEREO_FRAME
            if available <= 0:
                return None
            raw = bytes(self._byte_buffer[:available])
            del self._byte_buffer[:available]

        return decode_stereo_frames(raw)

    def read_available_pairs(self, pair_count: int) -> Optional[np.ndarray]:
        return self._pop_stereo_frames(pair_count, exact=False)

    def _select_mono_channel(self, stereo: np.ndarray) -> np.ndarray:
        mono, used_mode = select_mono_channel(stereo, channel_mode=self.cfg.channel_mode)
        self.last_channel_used = used_mode
        return mono

    def _prepare_audio_frame(self, stereo: np.ndarray) -> np.ndarray:
        frame, used_mode = prepare_audio_window(
            stereo,
            channel_mode=self.cfg.channel_mode,
            sample_shift_bits=self.cfg.sample_shift_bits,
            remove_dc=self.cfg.remove_dc,
        )
        self.last_channel_used = used_mode
        self.last_audio_frame = frame.astype(np.float32, copy=True)
        return frame

    def start(self) -> None:
        resolved_device = resolve_audio_device(self.cfg.device)
        self.cfg.device = resolved_device
        cmd = build_capture_cmd(
            resolved_device,
            self.cfg.sample_rate,
            backend=self.cfg.capture_backend,
            capture_binary=self.cfg.capture_binary,
        )
        self._byte_buffer.clear()
        self.last_channel_used = CHANNEL_MODE_LEFT
        self.last_audio_frame.fill(0.0)
        self.last_frame_bfpexp = 0
        self.last_frame_had_explicit_bfpexp = False
        self.last_frame_missing_bins = tuple()
        self._proc = start_capture_process(
            resolved_device,
            self.cfg.sample_rate,
            backend=self.cfg.capture_backend,
            capture_binary=self.cfg.capture_binary,
        )
        print("Starting:", " ".join(cmd), flush=True)

    def stop(self) -> None:
        if self._proc is not None:
            stop_process(self._proc)
            self._proc = None
        self._byte_buffer.clear()
        self.last_frame_bfpexp = 0
        self.last_frame_had_explicit_bfpexp = False
        self.last_frame_missing_bins = tuple()

    def read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        stereo = self._pop_stereo_frames(self.cfg.frame_bins, exact=True)
        if stereo is None:
            return None

        audio_frame = self._prepare_audio_frame(stereo)
        fft_full, fft_useful = compute_fft_magnitude(
            audio_frame,
            frame_bins=self.cfg.frame_bins,
            useful_bins=self.cfg.useful_bins,
            window=self._window,
        )

        mel = self._mel_filter @ fft_full
        mel = np.log(mel + 1e-9)
        mfcc = self._dct_matrix @ mel

        return fft_useful.astype(np.float32, copy=False), mfcc.astype(np.float32, copy=False)
