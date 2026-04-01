import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from .i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_RATE_HZ,
        TaggedI2SRealigner,
        build_arecord_cmd,
        resolve_audio_device,
        start_arecord_process,
        stop_process,
    )
    from .spectral_features import build_dct_matrix, build_mel_filter
except ImportError:
    from i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_RATE_HZ,
        TaggedI2SRealigner,
        build_arecord_cmd,
        resolve_audio_device,
        start_arecord_process,
        stop_process,
    )
    from spectral_features import build_dct_matrix, build_mel_filter

try:
    import gpiod  # type: ignore
except ImportError:  # pragma: no cover - optional dependency on target device
    gpiod = None


@dataclass
class FFTAdapterConfig:
    device: str = AUTO_AUDIO_DEVICE
    sample_rate: int = DEFAULT_CAPTURE_RATE_HZ
    frame_bins: int = 512
    useful_bins: int = 256
    gpio_chip: str = "/dev/gpiochip0"
    bfpexp_flag_line: Optional[int] = None
    done_line: Optional[int] = None
    flag_active_high: bool = True
    done_pulse_seconds: float = 0.0005
    handshake_timeout_seconds: float = 1.0
    wait_for_flag_falling_edge: bool = True
    use_i2s_tags: bool = False
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
        if not 1 <= self.payload_bits <= 31:
            raise ValueError("payload_bits must be between 1 and 31")
        if not 0 <= self.tag_shift <= 31:
            raise ValueError("tag_shift must be between 0 and 31")
        if self.tag_mask <= 0:
            raise ValueError("tag_mask must be positive")
        tag_width = int(self.tag_mask).bit_length()
        if (self.tag_shift + tag_width) > 32:
            raise ValueError("tag field must fit inside a 32-bit I2S word")
        if self.use_i2s_tags and self.payload_bits > self.tag_shift:
            raise ValueError("payload_bits must not overlap the tag field when use_i2s_tags is enabled")
        if (
            self.bfpexp_flag_line is not None
            and self.done_line is not None
            and self.bfpexp_flag_line == self.done_line
        ):
            raise ValueError("bfpexp_flag_line and done_line must be different GPIO lines")
        if self.handshake_timeout_seconds <= 0.0:
            raise ValueError("handshake_timeout_seconds must be positive")
        if self.done_pulse_seconds < 0.0:
            raise ValueError("done_pulse_seconds must be non-negative")


class FPGAFFTReceiver:
    def __init__(self, cfg: FFTAdapterConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._bytes_per_pair = 8  # real(int32) + imag(int32)
        self._frame_bytes = self.cfg.frame_bins * self._bytes_per_pair
        # Poll at least one full FFT frame per read so Python/GPIO overhead
        # does not force the ALSA capture side to run near the overrun limit.
        self._poll_pairs = max(64, self.cfg.frame_bins)
        self._poll_bytes = self._poll_pairs * self._bytes_per_pair
        self._byte_buffer = bytearray()
        self._line_request = None
        self._bfpexp_line = None
        self._done_line = None
        self._gpio_api = None
        self._gpio_chip = None
        self._tagged_realigner = TaggedI2SRealigner()

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
        if self.cfg.bfpexp_flag_line is None and self.cfg.done_line is None:
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
        line_module = getattr(gpiod, "line", gpiod)
        settings = {}

        if self.cfg.bfpexp_flag_line is not None:
            settings[self.cfg.bfpexp_flag_line] = gpiod.LineSettings(
                direction=line_module.Direction.INPUT,
            )

        if self.cfg.done_line is not None:
            settings[self.cfg.done_line] = gpiod.LineSettings(
                direction=line_module.Direction.OUTPUT,
                output_value=line_module.Value.INACTIVE,
            )

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
        self._bfpexp_line = self.cfg.bfpexp_flag_line
        self._done_line = self.cfg.done_line

    def _setup_gpio_v1(self, chip: object) -> None:
        self._gpio_api = "v1"

        if self.cfg.bfpexp_flag_line is not None:
            line = chip.get_line(self.cfg.bfpexp_flag_line)
            line.request(consumer="fpga_fft_receiver", type=gpiod.LINE_REQ_DIR_IN)
            self._bfpexp_line = line

        if self.cfg.done_line is not None:
            line = chip.get_line(self.cfg.done_line)
            line.request(consumer="fpga_fft_receiver", type=gpiod.LINE_REQ_DIR_OUT)
            line.set_value(0)
            self._done_line = line

    def _teardown_gpio(self) -> None:
        if self._line_request is not None:
            release = getattr(self._line_request, "release", None)
            if callable(release):
                release()
            self._line_request = None

        for line in (self._bfpexp_line, self._done_line):
            release = getattr(line, "release", None)
            if callable(release):
                release()

        close = getattr(self._gpio_chip, "close", None)
        if callable(close):
            close()

        self._bfpexp_line = None
        self._done_line = None
        self._gpio_api = None
        self._gpio_chip = None

    def _read_flag_active(self) -> bool:
        if self._bfpexp_line is None:
            return False

        if self._gpio_api == "v1":
            value = self._bfpexp_line.get_value()
            is_high = bool(value)
        else:
            if self._line_request is None:
                return False
            value = self._line_request.get_value(self._bfpexp_line)
            line_module = getattr(gpiod, "line", gpiod)
            is_high = value == line_module.Value.ACTIVE

        return is_high if self.cfg.flag_active_high else (not is_high)

    def _set_done(self, active: bool) -> None:
        if self._done_line is None:
            return

        if self._gpio_api == "v1":
            self._done_line.set_value(1 if active else 0)
            return

        if self._line_request is None:
            return
        line_module = getattr(gpiod, "line", gpiod)
        out = line_module.Value.ACTIVE if active else line_module.Value.INACTIVE
        self._line_request.set_value(self._done_line, out)

    def _pulse_done(self) -> None:
        if self._done_line is None:
            return
        self._set_done(True)
        time.sleep(max(0.0, self.cfg.done_pulse_seconds))
        self._set_done(False)

    def _fill_buffer(self, min_bytes: int) -> bool:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Receiver not started")

        while len(self._byte_buffer) < min_bytes:
            chunk = self._proc.stdout.read(max(self._poll_bytes, min_bytes - len(self._byte_buffer)))
            if not chunk:
                break
            self._byte_buffer.extend(chunk)

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
        if self._bfpexp_line is None:
            return None
        return self._read_flag_active()

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
        if not self.cfg.require_bfpexp_before_fft:
            return True
        # In tagged streams that wait for RPi DONE before emitting the next BFPEXP,
        # insisting on BFPEXP for the very first decoded frame can deadlock startup
        # if software attaches while a FFT burst is already in flight.
        return self.cfg.done_line is not None

    def _wait_for_fft_window(self) -> bool:
        if self.cfg.use_i2s_tags:
            return True
        if self._bfpexp_line is None:
            return True
        if self._gpio_api == "v2" and self._line_request is None:
            return True

        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Receiver not started")

        deadline = time.monotonic() + max(0.001, self.cfg.handshake_timeout_seconds)
        previous_active = self._read_flag_active()

        # Drain audio while watching GPIO so the capture pointer stays near real time.
        while time.monotonic() < deadline:
            pairs = self._pop_pairs(self._poll_pairs, exact=False)
            if pairs is None:
                return False

            current_active = self._read_flag_active()
            if self.cfg.wait_for_flag_falling_edge:
                if previous_active and (not current_active):
                    return True
            else:
                if not current_active:
                    return True
            previous_active = current_active

        return False

    def _read_frame_from_i2s_tags(self) -> Optional[np.ndarray]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Receiver not started")

        deadline = time.monotonic() + max(0.001, self.cfg.handshake_timeout_seconds)
        fft_pairs = []
        waiting_for_start = True
        bfpexp_seen = False

        while time.monotonic() < deadline:
            pairs = self._pop_pairs(self._poll_pairs, exact=False)
            if pairs is None:
                return None
            pairs = self._tagged_realigner.push_pairs(pairs)
            if pairs.size == 0:
                continue
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

                # Any non-FFT tag after frame start breaks the partial frame.
                # Discard the partial data and keep scanning the current chunk so
                # idle/null padding is ignored and a fresh BFPEXP can resync us.
                fft_pairs.clear()
                waiting_for_start = True
                bfpexp_seen = (kind == "bfpexp")

        return None

    def start(self) -> None:
        resolved_device = resolve_audio_device(self.cfg.device)
        self.cfg.device = resolved_device
        cmd = build_arecord_cmd(resolved_device, self.cfg.sample_rate)
        self._byte_buffer.clear()
        self._tagged_realigner.reset()
        try:
            self._proc = start_arecord_process(resolved_device, self.cfg.sample_rate)
            self._setup_gpio()
        except Exception:
            if self._proc is not None:
                stop_process(self._proc)
                self._proc = None
            self._teardown_gpio()
            raise
        print("Starting:", " ".join(cmd), flush=True)

    def stop(self) -> None:
        self._set_done(False)
        if self._proc is not None:
            stop_process(self._proc)
            self._proc = None
        self._byte_buffer.clear()
        self._tagged_realigner.reset()
        self._teardown_gpio()

    def read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.cfg.use_i2s_tags:
            pairs = self._read_frame_from_i2s_tags()
            if pairs is None:
                return None
        else:
            if not self._wait_for_fft_window():
                return None

            pairs = self._pop_pairs(self.cfg.frame_bins, exact=True)
            if pairs is None:
                return None

        real = pairs[:, 0].astype(np.float32)
        imag = pairs[:, 1].astype(np.float32)

        # Magnitude spectrum from complex bins streamed by FPGA.
        fft_mag = np.sqrt(real * real + imag * imag)
        fft_useful = fft_mag[: self.cfg.useful_bins]

        mel = self.mel_filter @ fft_useful
        mel = np.log(mel + 1e-9)
        mfcc = self._dct_matrix @ mel

        self._pulse_done()

        return fft_useful.astype(np.float32, copy=False), mfcc.astype(np.float32, copy=False)
