import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
from scipy.fftpack import dct

try:
    import gpiod  # type: ignore
except ImportError:  # pragma: no cover - optional dependency on target device
    gpiod = None


@dataclass
class FFTAdapterConfig:
    device: str = "hw:2,0"
    sample_rate: int = 48000
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


class FPGAFFTReceiver:
    def __init__(self, cfg: FFTAdapterConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._bytes_per_pair = 8  # real(int32) + imag(int32)
        self._frame_bytes = self.cfg.frame_bins * self._bytes_per_pair
        self._poll_pairs = 64
        self._poll_bytes = self._poll_pairs * self._bytes_per_pair
        self._line_request = None
        self._bfpexp_line = None
        self._done_line = None

        payload_mask = (1 << self.cfg.payload_bits) - 1
        self._payload_mask = payload_mask
        self._payload_sign_bit = 1 << (self.cfg.payload_bits - 1)

        n_fft = max(2, 2 * (self.cfg.useful_bins - 1))
        self.mel_filter = librosa.filters.mel(
            sr=self.cfg.sample_rate,
            n_fft=n_fft,
            n_mels=32,
        ).astype(np.float32)

    def _setup_gpio(self) -> None:
        if self.cfg.bfpexp_flag_line is None and self.cfg.done_line is None:
            return
        if gpiod is None:
            raise RuntimeError(
                "GPIO handshake requested but python gpiod is not installed. "
                "Install python3-gpiod on the Raspberry Pi."
            )

        chip = gpiod.Chip(self.cfg.gpio_chip)
        settings = {}

        if self.cfg.bfpexp_flag_line is not None:
            settings[self.cfg.bfpexp_flag_line] = gpiod.LineSettings(
                direction=gpiod.line.Direction.INPUT,
            )

        if self.cfg.done_line is not None:
            settings[self.cfg.done_line] = gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT,
                output_value=gpiod.line.Value.INACTIVE,
            )

        self._line_request = chip.request_lines(
            consumer="fpga_fft_receiver",
            config=settings,
        )

        if self.cfg.bfpexp_flag_line is not None:
            self._bfpexp_line = self.cfg.bfpexp_flag_line
        if self.cfg.done_line is not None:
            self._done_line = self.cfg.done_line

    def _teardown_gpio(self) -> None:
        if self._line_request is not None:
            self._line_request.release()
            self._line_request = None
        self._bfpexp_line = None
        self._done_line = None

    def _read_flag_active(self) -> bool:
        if self._line_request is None or self._bfpexp_line is None:
            return False
        value = self._line_request.get_value(self._bfpexp_line)
        is_high = value == gpiod.line.Value.ACTIVE
        return is_high if self.cfg.flag_active_high else (not is_high)

    def _set_done(self, active: bool) -> None:
        if self._line_request is None or self._done_line is None:
            return
        out = gpiod.line.Value.ACTIVE if active else gpiod.line.Value.INACTIVE
        self._line_request.set_value(self._done_line, out)

    def _pulse_done(self) -> None:
        if self._done_line is None:
            return
        self._set_done(True)
        time.sleep(max(0.0, self.cfg.done_pulse_seconds))
        self._set_done(False)

    def _read_pairs_exact(self, pair_count: int) -> Optional[np.ndarray]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Receiver not started")

        needed = pair_count * self._bytes_per_pair
        raw = self._proc.stdout.read(needed)
        if len(raw) != needed:
            return None
        return np.frombuffer(raw, dtype=np.int32).reshape(-1, 2)

    def _decode_tagged_word(self, word: int) -> Tuple[int, int]:
        uword = int(np.uint32(word))
        tag = (uword >> self.cfg.tag_shift) & self.cfg.tag_mask
        payload = uword & self._payload_mask
        if payload & self._payload_sign_bit:
            payload -= 1 << self.cfg.payload_bits
        return tag, payload

    def _pair_kind_and_payload(self, pair: np.ndarray) -> Tuple[str, Tuple[int, int]]:
        tag_l, payload_l = self._decode_tagged_word(int(pair[0]))
        tag_r, payload_r = self._decode_tagged_word(int(pair[1]))

        if tag_l == self.cfg.tag_fft and tag_r == self.cfg.tag_fft:
            return "fft", (payload_l, payload_r)
        if tag_l == self.cfg.tag_bfpexp or tag_r == self.cfg.tag_bfpexp:
            return "bfpexp", (payload_l, payload_r)
        if tag_l == self.cfg.tag_idle and tag_r == self.cfg.tag_idle:
            return "idle", (payload_l, payload_r)
        return "other", (payload_l, payload_r)

    def _wait_for_fft_window(self) -> bool:
        if self.cfg.use_i2s_tags:
            return True
        if self._bfpexp_line is None or self._line_request is None:
            return True

        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Receiver not started")

        deadline = time.monotonic() + max(0.001, self.cfg.handshake_timeout_seconds)
        previous_active = self._read_flag_active()

        # Drain audio while watching GPIO so the capture pointer stays near real time.
        while time.monotonic() < deadline:
            if self._proc.stdout.read(self._poll_bytes) == b"":
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
            raw = self._proc.stdout.read(self._poll_bytes)
            if raw == b"":
                return None
            if len(raw) % self._bytes_per_pair != 0:
                continue

            pairs = np.frombuffer(raw, dtype=np.int32).reshape(-1, 2)
            for pair in pairs:
                kind, payload = self._pair_kind_and_payload(pair)

                if waiting_for_start:
                    if kind == "bfpexp":
                        bfpexp_seen = True
                        continue
                    if kind == "fft":
                        if self.cfg.require_bfpexp_before_fft and not bfpexp_seen:
                            continue
                        waiting_for_start = False
                        fft_pairs.append(payload)
                        if len(fft_pairs) >= self.cfg.frame_bins:
                            return np.asarray(fft_pairs, dtype=np.int32)
                        continue
                    continue

                # After frame start, count only FFT-tagged pairs.
                if kind != "fft":
                    # Frame broke early; force re-sync from a fresh BFPEXP->FFT transition.
                    return None

                fft_pairs.append(payload)
                if len(fft_pairs) >= self.cfg.frame_bins:
                    return np.asarray(fft_pairs, dtype=np.int32)

        return None

    @staticmethod
    def _build_arecord_cmd(device: str, sample_rate: int) -> list:
        return [
            "arecord",
            "-q",
            "-D",
            device,
            "-f",
            "S32_LE",
            "-c",
            "2",
            "-r",
            str(sample_rate),
            "-t",
            "raw",
        ]

    def start(self) -> None:
        cmd = self._build_arecord_cmd(self.cfg.device, self.cfg.sample_rate)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._setup_gpio()

    def stop(self) -> None:
        self._set_done(False)
        if self._proc is None:
            self._teardown_gpio()
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._teardown_gpio()

    def read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.cfg.use_i2s_tags:
            pairs = self._read_frame_from_i2s_tags()
            if pairs is None:
                return None
        else:
            if not self._wait_for_fft_window():
                return None

            pairs = self._read_pairs_exact(self.cfg.frame_bins)
            if pairs is None:
                return None

        real = pairs[:, 0].astype(np.float32)
        imag = pairs[:, 1].astype(np.float32)

        # Magnitude spectrum from complex bins streamed by FPGA.
        fft_mag = np.sqrt(real * real + imag * imag)
        fft_useful = fft_mag[: self.cfg.useful_bins]

        mel = self.mel_filter @ fft_useful
        mel = np.log(mel + 1e-9)
        mfcc = dct(mel, type=2, norm="ortho")[:13].astype(np.float32)

        self._pulse_done()

        return fft_useful.astype(np.float32), mfcc
