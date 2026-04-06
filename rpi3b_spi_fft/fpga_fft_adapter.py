import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

try:
    from .spi_stream import (
        AUTO_SPI_DEVICE,
        BYTES_PER_FFT_PAIR,
        DEFAULT_SPI_BITS_PER_WORD,
        DEFAULT_SPI_MAX_SPEED_HZ,
        DEFAULT_SPI_MODE,
        close_spi_device,
        open_spi_device,
        resolve_spi_device,
        transfer_exactly,
    )
    from .spectral_features import build_dct_matrix, build_mel_filter
except ImportError:
    from spi_stream import (
        AUTO_SPI_DEVICE,
        BYTES_PER_FFT_PAIR,
        DEFAULT_SPI_BITS_PER_WORD,
        DEFAULT_SPI_MAX_SPEED_HZ,
        DEFAULT_SPI_MODE,
        close_spi_device,
        open_spi_device,
        resolve_spi_device,
        transfer_exactly,
    )
    from spectral_features import build_dct_matrix, build_mel_filter

try:
    import gpiod  # type: ignore
except ImportError:  # pragma: no cover - optional dependency on target device
    gpiod = None


@dataclass
class FFTAdapterConfig:
    device: str = AUTO_SPI_DEVICE
    sample_rate: int = 48000
    frame_bins: int = 512
    useful_bins: int = 256
    spi_max_speed_hz: int = DEFAULT_SPI_MAX_SPEED_HZ
    spi_mode: int = DEFAULT_SPI_MODE
    spi_bits_per_word: int = DEFAULT_SPI_BITS_PER_WORD
    gpio_chip: str = "/dev/gpiochip0"
    window_ready_line: Optional[int] = None
    handshake_timeout_seconds: float = 1.0
    bfpexp_hold_frames: int = 1
    use_word_tags: bool = True
    tag_shift: int = 30
    tag_mask: int = 0x3
    payload_bits: int = 18
    tag_idle: int = 0
    tag_bfpexp: int = 1
    tag_fft: int = 2
    require_bfpexp_before_fft: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_bins <= 0:
            raise ValueError("frame_bins must be positive")
        if not 2 <= self.useful_bins <= self.frame_bins:
            raise ValueError("Expected 2 <= useful_bins <= frame_bins")
        if self.spi_max_speed_hz <= 0:
            raise ValueError("spi_max_speed_hz must be positive")
        if not 0 <= self.spi_mode <= 3:
            raise ValueError("spi_mode must be between 0 and 3")
        if self.spi_bits_per_word <= 0:
            raise ValueError("spi_bits_per_word must be positive")
        if self.handshake_timeout_seconds <= 0.0:
            raise ValueError("handshake_timeout_seconds must be positive")
        if self.bfpexp_hold_frames < 1:
            raise ValueError("bfpexp_hold_frames must be >= 1")
        if not 1 <= self.payload_bits <= 31:
            raise ValueError("payload_bits must be between 1 and 31")
        if not 0 <= self.tag_shift <= 31:
            raise ValueError("tag_shift must be between 0 and 31")
        if self.tag_mask <= 0:
            raise ValueError("tag_mask must be positive")
        tag_width = int(self.tag_mask).bit_length()
        if (self.tag_shift + tag_width) > 32:
            raise ValueError("tag field must fit inside a 32-bit word")
        if self.use_word_tags and self.payload_bits > self.tag_shift:
            raise ValueError("payload_bits must not overlap the tag field when tags are enabled")


class FPGAFFTReceiver:
    def __init__(self, cfg: FFTAdapterConfig):
        self.cfg = cfg
        self._spi: Optional[Any] = None
        self._bytes_per_pair = BYTES_PER_FFT_PAIR
        self._transaction_pairs = (
            self.cfg.frame_bins + self.cfg.bfpexp_hold_frames
            if self.cfg.use_word_tags
            else self.cfg.frame_bins
        )
        self._transaction_bytes = self._transaction_pairs * self._bytes_per_pair
        self._poll_pairs = max(64, self._transaction_pairs)
        self._poll_bytes = self._poll_pairs * self._bytes_per_pair
        self._byte_buffer = bytearray()
        self._line_request = None
        self._window_ready_line = None
        self._gpio_api = None
        self._gpio_chip = None

        payload_mask = (1 << self.cfg.payload_bits) - 1
        self._payload_mask = payload_mask
        self._payload_sign_bit = 1 << (self.cfg.payload_bits - 1)

        n_fft = 2 * (self.cfg.useful_bins - 1)
        self.mel_filter = build_mel_filter(
            sample_rate=self.cfg.sample_rate,
            n_fft=n_fft,
            n_mels=32,
        )
        self._dct_matrix = build_dct_matrix(input_size=32, output_size=13)

    def _setup_gpio(self) -> None:
        if self.cfg.window_ready_line is None:
            return
        if gpiod is None:
            raise RuntimeError(
                "GPIO handshake requested but python gpiod is not installed. "
                "Install python3-libgpiod on the Raspberry Pi."
            )

        chip = gpiod.Chip(self.cfg.gpio_chip)
        self._gpio_chip = chip

        if hasattr(gpiod, "LineSettings"):
            self._setup_gpio_v2(chip)
            return

        self._setup_gpio_v1(chip)

    def _setup_gpio_v2(self, chip: object) -> None:
        assert self.cfg.window_ready_line is not None
        line_module = getattr(gpiod, "line", gpiod)
        settings = {
            self.cfg.window_ready_line: gpiod.LineSettings(
                direction=line_module.Direction.INPUT,
            )
        }

        if hasattr(chip, "request_lines"):
            self._line_request = chip.request_lines(
                consumer="fpga_fft_receiver",
                config=settings,
            )
        else:
            self._line_request = gpiod.request_lines(
                self.cfg.gpio_chip,
                consumer="fpga_fft_receiver",
                config=settings,
            )

        self._gpio_api = "v2"
        self._window_ready_line = self.cfg.window_ready_line

    def _setup_gpio_v1(self, chip: object) -> None:
        assert self.cfg.window_ready_line is not None
        line = chip.get_line(self.cfg.window_ready_line)
        line.request(consumer="fpga_fft_receiver", type=gpiod.LINE_REQ_DIR_IN)
        self._gpio_api = "v1"
        self._window_ready_line = line

    def _teardown_gpio(self) -> None:
        if self._line_request is not None:
            release = getattr(self._line_request, "release", None)
            if callable(release):
                release()
            self._line_request = None

        release = getattr(self._window_ready_line, "release", None)
        if callable(release):
            release()

        close = getattr(self._gpio_chip, "close", None)
        if callable(close):
            close()

        self._window_ready_line = None
        self._gpio_api = None
        self._gpio_chip = None

    def _read_window_ready(self) -> bool:
        if self._window_ready_line is None:
            return True

        if self._gpio_api == "v1":
            value = self._window_ready_line.get_value()
            return bool(value)

        if self._line_request is None:
            return False
        value = self._line_request.get_value(self._window_ready_line)
        line_module = getattr(gpiod, "line", gpiod)
        return value == line_module.Value.ACTIVE

    def _wait_for_window_ready(self) -> bool:
        if self.cfg.window_ready_line is None:
            return True

        deadline = time.monotonic() + max(0.001, self.cfg.handshake_timeout_seconds)
        while time.monotonic() < deadline:
            if self._read_window_ready():
                return True
            time.sleep(0.0005)
        return False

    def _capture_transaction(self) -> bool:
        if self._spi is None:
            return False

        if not self._wait_for_window_ready():
            return False

        raw = transfer_exactly(self._spi, self._transaction_bytes)
        if not raw:
            return False
        self._byte_buffer.extend(raw)
        return True

    def _fill_buffer(self, min_bytes: int) -> bool:
        while len(self._byte_buffer) < min_bytes:
            if not self._capture_transaction():
                break
        return len(self._byte_buffer) >= min_bytes

    def _pop_pairs(self, pair_count: int, exact: bool) -> Optional[np.ndarray]:
        if pair_count <= 0:
            return np.empty((0, 2), dtype=np.int32)

        needed = pair_count * self._bytes_per_pair
        if exact:
            if not self._fill_buffer(needed):
                return None
            raw = bytes(self._byte_buffer[:needed])
            del self._byte_buffer[:needed]
        else:
            if not self._fill_buffer(self._bytes_per_pair):
                return None
            available = min(len(self._byte_buffer), needed)
            available -= available % self._bytes_per_pair
            if available <= 0:
                return None
            raw = bytes(self._byte_buffer[:available])
            del self._byte_buffer[:available]

        if not raw:
            return None
        return np.frombuffer(raw, dtype=np.int32).reshape(-1, 2)

    def read_available_pairs(self, pair_count: int) -> Optional[np.ndarray]:
        return self._pop_pairs(pair_count, exact=False)

    def read_flag_state(self) -> Optional[bool]:
        if self.cfg.window_ready_line is None:
            return None
        return self._read_window_ready()

    def _decode_tagged_word(self, word: int) -> Tuple[int, int]:
        uword = int(word) & 0xFFFFFFFF
        tag = (uword >> self.cfg.tag_shift) & self.cfg.tag_mask
        payload = uword & self._payload_mask
        if payload & self._payload_sign_bit:
            payload -= 1 << self.cfg.payload_bits
        return tag, payload

    def _push_pairs_back(self, pairs: np.ndarray) -> None:
        if pairs.size == 0:
            return
        self._byte_buffer[:0] = np.asarray(pairs, dtype=np.int32).tobytes()

    def _pair_kind_and_payload(self, pair: np.ndarray) -> Tuple[str, Tuple[int, int]]:
        tag_l, payload_l = self._decode_tagged_word(int(pair[0]))
        tag_r, payload_r = self._decode_tagged_word(int(pair[1]))

        if tag_l != tag_r:
            return "other", (payload_l, payload_r)
        if tag_l == self.cfg.tag_fft:
            return "fft", (payload_l, payload_r)
        if tag_l == self.cfg.tag_bfpexp:
            return "bfpexp", (payload_l, payload_r)
        if tag_l == self.cfg.tag_idle:
            return "idle", (payload_l, payload_r)
        return "other", (payload_l, payload_r)

    def _allow_tagged_fft_start_without_bfpexp(self) -> bool:
        return not self.cfg.require_bfpexp_before_fft

    def _read_frame_from_word_tags(self) -> Optional[np.ndarray]:
        deadline = time.monotonic() + max(0.001, self.cfg.handshake_timeout_seconds)
        fft_pairs = []
        waiting_for_start = True
        bfpexp_seen = False

        while time.monotonic() < deadline:
            pairs = self._pop_pairs(self._poll_pairs, exact=False)
            if pairs is None:
                return None
            for idx, pair in enumerate(pairs):
                kind, payload = self._pair_kind_and_payload(pair)

                if waiting_for_start:
                    if kind == "idle":
                        continue
                    if kind == "bfpexp":
                        bfpexp_seen = True
                        continue
                    if kind == "fft":
                        if (not bfpexp_seen) and (not self._allow_tagged_fft_start_without_bfpexp()):
                            continue
                        waiting_for_start = False
                        fft_pairs.append(payload)
                        if len(fft_pairs) >= self.cfg.frame_bins:
                            self._push_pairs_back(pairs[idx + 1 :])
                            return np.asarray(fft_pairs, dtype=np.int32)
                        continue
                    continue

                if kind == "fft":
                    fft_pairs.append(payload)
                    if len(fft_pairs) >= self.cfg.frame_bins:
                        self._push_pairs_back(pairs[idx + 1 :])
                        return np.asarray(fft_pairs, dtype=np.int32)
                    continue

                fft_pairs.clear()
                waiting_for_start = True
                bfpexp_seen = (kind == "bfpexp")

        return None

    def start(self) -> None:
        resolved_device = resolve_spi_device(self.cfg.device)
        self.cfg.device = resolved_device
        self._byte_buffer.clear()
        try:
            self._spi = open_spi_device(
                resolved_device,
                max_speed_hz=self.cfg.spi_max_speed_hz,
                mode=self.cfg.spi_mode,
                bits_per_word=self.cfg.spi_bits_per_word,
            )
            self._setup_gpio()
        except Exception:
            if self._spi is not None:
                close_spi_device(self._spi)
                self._spi = None
            self._teardown_gpio()
            raise

        print(
            "Starting SPI capture:",
            resolved_device,
            f"mode={self.cfg.spi_mode}",
            f"max_speed_hz={self.cfg.spi_max_speed_hz}",
            flush=True,
        )

    def stop(self) -> None:
        if self._spi is not None:
            close_spi_device(self._spi)
            self._spi = None
        self._byte_buffer.clear()
        self._teardown_gpio()

    def read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.cfg.use_word_tags:
            pairs = self._read_frame_from_word_tags()
            if pairs is None:
                return None
        else:
            pairs = self._pop_pairs(self.cfg.frame_bins, exact=True)
            if pairs is None:
                return None

        real = pairs[:, 0].astype(np.float32)
        imag = pairs[:, 1].astype(np.float32)

        fft_mag = np.sqrt(real * real + imag * imag)
        fft_useful = fft_mag[: self.cfg.useful_bins]

        mel = self.mel_filter @ fft_useful
        mel = np.log(mel + 1e-9)
        mfcc = self._dct_matrix @ mel

        return fft_useful.astype(np.float32, copy=False), mfcc.astype(np.float32, copy=False)
