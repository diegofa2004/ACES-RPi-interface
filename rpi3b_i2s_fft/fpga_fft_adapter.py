import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from .i2s_stream import (
        AUTO_AUDIO_DEVICE,
        CAPTURE_BACKEND_NATIVE,
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
        TaggedI2SRealigner,
        classify_tagged_i2s_pair,
        decode_tagged_i2s_word,
        helper_tagged_realign_enabled,
        parse_capture_telemetry_line,
        resolve_capture_command,
        resolve_audio_device,
        start_capture_process,
        stop_process,
    )
    from .spectral_features import build_dct_matrix, build_mel_filter
except ImportError:
    from i2s_stream import (
        AUTO_AUDIO_DEVICE,
        CAPTURE_BACKEND_NATIVE,
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
        TaggedI2SRealigner,
        classify_tagged_i2s_pair,
        decode_tagged_i2s_word,
        helper_tagged_realign_enabled,
        parse_capture_telemetry_line,
        resolve_capture_command,
        resolve_audio_device,
        start_capture_process,
        stop_process,
    )
    from spectral_features import build_dct_matrix, build_mel_filter

try:
    import gpiod  # type: ignore
except ImportError:  # pragma: no cover - optional dependency on target device
    gpiod = None


DEFAULT_BFPEXP_HOLD_PAIRS = 128
DEFAULT_TAG_LOSS_TOLERANCE_PAIRS = 3


@dataclass
class FFTAdapterConfig:
    device: str = AUTO_AUDIO_DEVICE
    sample_rate: int = DEFAULT_CAPTURE_RATE_HZ
    frame_bins: int = 512
    useful_bins: int = 256
    capture_backend: str = "auto"
    capture_binary: Optional[str] = None
    capture_telemetry: bool = False
    capture_realign_initial_word_skip: int = 0
    capture_realign_swap_channels: bool = False
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
    apply_bfpexp: bool = True
    require_bfpexp_before_fft: bool = True
    bfpexp_pairs_required: int = DEFAULT_BFPEXP_HOLD_PAIRS
    loss_tolerance_pairs: int = DEFAULT_TAG_LOSS_TOLERANCE_PAIRS

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.capture_realign_initial_word_skip < 0:
            raise ValueError("capture_realign_initial_word_skip must be non-negative")
        if self.frame_bins <= 0:
            raise ValueError("frame_bins must be positive")
        if self.frame_bins > self.fft_packet_index_base:
            raise ValueError("frame_bins must fit inside the FFT packet-index range")
        if not 2 <= self.useful_bins <= self.frame_bins:
            raise ValueError("Expected 2 <= useful_bins <= frame_bins")
        if not 1 <= self.payload_bits <= 31:
            raise ValueError("payload_bits must be between 1 and 31")
        if not 0 <= self.tag_shift <= 31:
            raise ValueError("tag_shift must be between 0 and 31")
        if not 0 <= self.packet_index_shift <= 31:
            raise ValueError("packet_index_shift must be between 0 and 31")
        if not 1 <= self.packet_index_bits <= 31:
            raise ValueError("packet_index_bits must be between 1 and 31")
        if self.tag_mask <= 0:
            raise ValueError("tag_mask must be positive")
        tag_width = int(self.tag_mask).bit_length()
        if (self.tag_shift + tag_width) > 32:
            raise ValueError("tag field must fit inside a 32-bit I2S word")
        if (self.packet_index_shift + self.packet_index_bits) > 32:
            raise ValueError("packet index field must fit inside a 32-bit I2S word")
        if self.packet_index_shift < (self.tag_shift + tag_width):
            raise ValueError("packet index field must not overlap the tag field")
        if self.use_i2s_tags and self.payload_bits > self.tag_shift:
            raise ValueError("payload_bits must not overlap the tag field when use_i2s_tags is enabled")
        if self.fft_packet_index_base != (1 << (self.packet_index_bits - 1)):
            raise ValueError("fft_packet_index_base must match the MSB of the packet index field")
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
        if self.bfpexp_pairs_required <= 0:
            raise ValueError("bfpexp_pairs_required must be positive")
        if self.bfpexp_pairs_required > self.fft_packet_index_base:
            raise ValueError("bfpexp_pairs_required must fit inside the BFPEXP packet-index range")
        if self.loss_tolerance_pairs < 0:
            raise ValueError("loss_tolerance_pairs must be non-negative")


class FPGAFFTReceiver:
    def __init__(self, cfg: FFTAdapterConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._bytes_per_pair = 8  # real(int32) + imag(int32)
        self._frame_bytes = self.cfg.frame_bins * self._bytes_per_pair
        self._helper_fft_header_words = 4
        self._helper_fft_frame_bytes = (self._helper_fft_header_words * 4) + self._frame_bytes
        # Keep polling granularity close to one FFT frame so live tools react
        # promptly while still avoiding pathological tiny reads.
        self._poll_pairs = max(512, self.cfg.frame_bins)
        self._poll_bytes = self._poll_pairs * self._bytes_per_pair
        self._byte_buffer = bytearray()
        self._line_request = None
        self._bfpexp_line = None
        self._done_line = None
        self._gpio_api = None
        self._gpio_chip = None
        self._tagged_realigner = TaggedI2SRealigner(
            packet_index_shift=self.cfg.packet_index_shift,
            packet_index_bits=self.cfg.packet_index_bits,
            tag_shift=self.cfg.tag_shift,
            tag_mask=self.cfg.tag_mask,
            payload_bits=self.cfg.payload_bits,
            tag_idle=self.cfg.tag_idle,
            tag_bfpexp=self.cfg.tag_bfpexp,
            tag_fft=self.cfg.tag_fft,
            fft_packet_index_base=self.cfg.fft_packet_index_base,
            confirm_pairs=min(64, max(4, self.cfg.frame_bins)),
            validate_pairs=min(64, max(4, self.cfg.frame_bins)),
            preferred_swap_channels=True,
        )
        self._tagged_pair_buffer = np.empty((0, 2), dtype=np.int32)

        payload_mask = (1 << self.cfg.payload_bits) - 1
        self._payload_mask = payload_mask
        self._payload_sign_bit = 1 << (self.cfg.payload_bits - 1)
        self.last_frame_bfpexp = 0
        self.last_frame_had_explicit_bfpexp = False
        self.last_frame_missing_bins: tuple[int, ...] = tuple()
        self._captured_frame_count = 0
        self._capture_telemetry_lock = threading.Lock()
        self._capture_telemetry_events: list[dict[str, object]] = []
        self._stderr_thread: Optional[threading.Thread] = None
        self._helper_handles_tagged_alignment = False
        self._helper_decodes_tagged_fft_frames = False

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

    def _capture_stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return

        while True:
            line = proc.stderr.readline()
            if not line:
                break
            event = parse_capture_telemetry_line(line)
            if event is None:
                continue
            with self._capture_telemetry_lock:
                self._capture_telemetry_events.append(event)

    def drain_capture_telemetry_events(self) -> list[dict[str, object]]:
        with self._capture_telemetry_lock:
            if not self._capture_telemetry_events:
                return []
            events = list(self._capture_telemetry_events)
            self._capture_telemetry_events.clear()
        return events

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

    def _decode_tagged_word(self, word: int) -> Tuple[int, int, int]:
        decoded = decode_tagged_i2s_word(
            int(word),
            tag_shift=self.cfg.tag_shift,
            tag_mask=self.cfg.tag_mask,
            payload_bits=self.cfg.payload_bits,
            packet_index_shift=self.cfg.packet_index_shift,
            packet_index_bits=self.cfg.packet_index_bits,
        )
        return int(decoded["tag"]), int(decoded["packet_index"]), int(decoded["payload"])

    def _push_pairs_back(self, pairs: np.ndarray) -> None:
        if pairs.size == 0:
            return
        pair_array = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        if self.cfg.use_i2s_tags:
            if self._tagged_pair_buffer.size == 0:
                self._tagged_pair_buffer = pair_array.copy()
            else:
                self._tagged_pair_buffer = np.concatenate((pair_array, self._tagged_pair_buffer), axis=0)
            return
        self._byte_buffer[:0] = pair_array.tobytes()

    def _pair_kind_packet_index_and_payload(
        self,
        pair: np.ndarray,
    ) -> Tuple[str, int, Tuple[int, int]]:
        tag_l, packet_index_l, payload_l = self._decode_tagged_word(int(pair[0]))
        tag_r, packet_index_r, payload_r = self._decode_tagged_word(int(pair[1]))
        kind = classify_tagged_i2s_pair(
            tag_l,
            tag_r,
            left_packet_index=packet_index_l,
            right_packet_index=packet_index_r,
            tag_idle=self.cfg.tag_idle,
            tag_bfpexp=self.cfg.tag_bfpexp,
            tag_fft=self.cfg.tag_fft,
            fft_packet_index_base=self.cfg.fft_packet_index_base,
        )
        return kind, packet_index_l, (payload_l, payload_r)

    def _decode_tagged_pair_batch(
        self,
        pairs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair_array = np.asarray(pairs, dtype=np.int32)
        if pair_array.size == 0:
            return (
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),
            )
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            pair_array = pair_array.reshape(-1, 2)

        pair_u32 = pair_array.astype(np.uint32, copy=False)
        left = pair_u32[:, 0]
        right = pair_u32[:, 1]

        packet_index_mask = np.uint32((1 << self.cfg.packet_index_bits) - 1)
        packet_index_l_u32 = (left >> np.uint32(self.cfg.packet_index_shift)) & packet_index_mask
        packet_index_r_u32 = (right >> np.uint32(self.cfg.packet_index_shift)) & packet_index_mask
        tags_l = (left >> np.uint32(self.cfg.tag_shift)) & np.uint32(self.cfg.tag_mask)
        tags_r = (right >> np.uint32(self.cfg.tag_shift)) & np.uint32(self.cfg.tag_mask)

        payload_l = np.bitwise_and(left, np.uint32(self._payload_mask)).astype(np.int32, copy=False)
        payload_r = np.bitwise_and(right, np.uint32(self._payload_mask)).astype(np.int32, copy=False)
        payload_l = np.where(
            (payload_l & self._payload_sign_bit) != 0,
            payload_l - (1 << self.cfg.payload_bits),
            payload_l,
        ).astype(np.int32, copy=False)
        payload_r = np.where(
            (payload_r & self._payload_sign_bit) != 0,
            payload_r - (1 << self.cfg.payload_bits),
            payload_r,
        ).astype(np.int32, copy=False)

        tags_match = tags_l == tags_r
        packet_index_match = packet_index_l_u32 == packet_index_r_u32
        valid_pairs = tags_match & packet_index_match

        kind_codes = np.zeros(pair_array.shape[0], dtype=np.int8)
        idle_mask = valid_pairs & (tags_l == self.cfg.tag_idle) & (packet_index_l_u32 == 0)
        bfpexp_mask = (
            valid_pairs
            & (tags_l == self.cfg.tag_bfpexp)
            & (packet_index_l_u32 < self.cfg.fft_packet_index_base)
        )
        fft_mask = (
            valid_pairs
            & (tags_l == self.cfg.tag_fft)
            & (packet_index_l_u32 >= self.cfg.fft_packet_index_base)
        )
        kind_codes[idle_mask] = 1
        kind_codes[bfpexp_mask] = 2
        kind_codes[fft_mask] = 3

        payloads = np.empty_like(pair_array)
        payloads[:, 0] = payload_l
        payloads[:, 1] = payload_r
        packet_indices = packet_index_l_u32.astype(np.int32, copy=False)
        return kind_codes, packet_indices, payloads

    def _allow_tagged_fft_start_without_bfpexp(self) -> bool:
        if not self.cfg.require_bfpexp_before_fft:
            return True
        # In tagged streams that wait for RPi DONE before emitting the next BFPEXP,
        # insisting on BFPEXP for the very first decoded frame can deadlock startup
        # if software attaches while a FFT burst is already in flight. After the
        # first completed frame, DONE should keep the transport aligned and BFPEXP
        # must be required again to avoid locking onto a mid-burst FFT payload.
        return self.cfg.done_line is not None and self._captured_frame_count == 0

    @staticmethod
    def _is_tolerable_loss_kind(kind: str) -> bool:
        return kind in ("tag_mismatch", "packet_index_mismatch", "unknown_tag", "idle")

    def _finalize_tagged_frame(
        self,
        frame_pairs: np.ndarray,
        received_bins: np.ndarray,
        frame_bfpexp: int,
        have_explicit_bfpexp: bool,
    ) -> Tuple[np.ndarray, int]:
        missing_bins = tuple(int(idx) for idx in np.flatnonzero(~received_bins))
        self.last_frame_bfpexp = int(frame_bfpexp)
        self.last_frame_had_explicit_bfpexp = bool(have_explicit_bfpexp)
        self.last_frame_missing_bins = missing_bins
        self._captured_frame_count += 1
        return frame_pairs.copy(), int(frame_bfpexp)

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

    def _read_frame_from_i2s_tags(self) -> Optional[Tuple[np.ndarray, int]]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Receiver not started")

        deadline = time.monotonic() + max(0.001, self.cfg.handshake_timeout_seconds)
        fft_pairs = np.zeros((self.cfg.frame_bins, 2), dtype=np.int32)
        received_bins = np.zeros(self.cfg.frame_bins, dtype=bool)
        received_count = 0
        waiting_for_start = True
        bfpexp_seen_indices: set[int] = set()
        bfpexp_gap_count = 0
        highest_bin_index_seen = -1
        current_bfpexp = int(self.last_frame_bfpexp)
        have_explicit_bfpexp = False
        required_bfpexp_pairs = self.cfg.bfpexp_pairs_required
        tolerated_losses = self.cfg.loss_tolerance_pairs

        while time.monotonic() < deadline:
            if self._tagged_pair_buffer.size != 0:
                pairs = self._tagged_pair_buffer
                self._tagged_pair_buffer = np.empty((0, 2), dtype=np.int32)
            else:
                pairs = self._pop_pairs(self._poll_pairs, exact=False)
                if pairs is None:
                    break
                if not self._helper_handles_tagged_alignment:
                    pairs = self._tagged_realigner.push_pairs(pairs)
                    if pairs.size == 0:
                        continue
            kind_codes, packet_indices, payloads = self._decode_tagged_pair_batch(pairs)
            for idx in range(pairs.shape[0]):
                kind_code = int(kind_codes[idx])
                packet_index = int(packet_indices[idx])
                payload = payloads[idx]

                if waiting_for_start:
                    if kind_code == 1:
                        if bfpexp_seen_indices and bfpexp_gap_count < tolerated_losses:
                            bfpexp_gap_count += 1
                        continue
                    if kind_code == 2:
                        current_bfpexp = int(payload[0])
                        have_explicit_bfpexp = True
                        if packet_index == 0:
                            bfpexp_seen_indices = {0}
                            bfpexp_gap_count = 0
                        elif 0 <= packet_index < required_bfpexp_pairs:
                            bfpexp_seen_indices.add(packet_index)
                        continue
                    if kind_code == 0:
                        if bfpexp_seen_indices and bfpexp_gap_count < tolerated_losses:
                            bfpexp_gap_count += 1
                        continue
                    if kind_code == 3:
                        bin_index = packet_index - self.cfg.fft_packet_index_base
                        if bin_index < 0 or bin_index >= self.cfg.frame_bins:
                            continue
                        full_bfpexp_preamble = (
                            0 in bfpexp_seen_indices
                            and (len(bfpexp_seen_indices) + bfpexp_gap_count) >= required_bfpexp_pairs
                        )
                        bootstrap_from_fft = (not full_bfpexp_preamble) and self._allow_tagged_fft_start_without_bfpexp()
                        if (not full_bfpexp_preamble) and (not bootstrap_from_fft):
                            continue
                        waiting_for_start = False
                        fft_pairs.fill(0)
                        received_bins[:] = False
                        received_count = 0
                        fft_pairs[bin_index, 0] = payload[0]
                        fft_pairs[bin_index, 1] = payload[1]
                        if not bool(received_bins[bin_index]):
                            received_bins[bin_index] = True
                            received_count += 1
                        highest_bin_index_seen = bin_index
                        if received_count >= self.cfg.frame_bins:
                            self._push_pairs_back(pairs[idx + 1 :])
                            return self._finalize_tagged_frame(
                                fft_pairs,
                                received_bins,
                                current_bfpexp,
                                have_explicit_bfpexp,
                            )
                        continue
                    continue

                if kind_code == 2:
                    self._push_pairs_back(pairs[idx:])
                    return self._finalize_tagged_frame(
                        fft_pairs,
                        received_bins,
                        current_bfpexp,
                        have_explicit_bfpexp,
                    )
                if kind_code == 3:
                    bin_index = packet_index - self.cfg.fft_packet_index_base
                    if bin_index < 0 or bin_index >= self.cfg.frame_bins:
                        continue
                    if bin_index < highest_bin_index_seen:
                        self._push_pairs_back(pairs[idx:])
                        return self._finalize_tagged_frame(
                            fft_pairs,
                            received_bins,
                            current_bfpexp,
                            have_explicit_bfpexp,
                        )
                    fft_pairs[bin_index, 0] = payload[0]
                    fft_pairs[bin_index, 1] = payload[1]
                    if not bool(received_bins[bin_index]):
                        received_bins[bin_index] = True
                        received_count += 1
                    highest_bin_index_seen = bin_index
                    if received_count >= self.cfg.frame_bins:
                        self._push_pairs_back(pairs[idx + 1 :])
                        return self._finalize_tagged_frame(
                            fft_pairs,
                            received_bins,
                            current_bfpexp,
                            have_explicit_bfpexp,
                        )
                    continue
                if kind_code == 1:
                    # The FPGA transport may insert tagged idle padding between valid
                    # FFT payload pairs. Those words are not payload loss and should
                    # not break the current frame.
                    continue
                if kind_code == 0:
                    continue

                self._push_pairs_back(pairs[idx + 1 :])
                return self._finalize_tagged_frame(
                    fft_pairs,
                    received_bins,
                    current_bfpexp,
                    have_explicit_bfpexp,
                )

        if bool(np.any(received_bins)):
            return self._finalize_tagged_frame(
                fft_pairs,
                received_bins,
                current_bfpexp,
                have_explicit_bfpexp,
            )
        return None

    def _read_frame_from_helper_tagged_fft(self) -> Optional[Tuple[np.ndarray, int]]:
        if not self._fill_buffer(self._helper_fft_frame_bytes):
            return None

        raw = bytes(self._byte_buffer[: self._helper_fft_frame_bytes])
        del self._byte_buffer[: self._helper_fft_frame_bytes]

        words = np.frombuffer(raw, dtype=np.int32)
        header = words[: self._helper_fft_header_words]
        payload = words[self._helper_fft_header_words :].reshape(self.cfg.frame_bins, 2)

        frame_bfpexp = int(header[0])
        flags = int(header[1])
        received_count = int(header[2])
        self.last_frame_bfpexp = frame_bfpexp
        self.last_frame_had_explicit_bfpexp = bool(flags & 0x1)
        # The helper currently reports only the number of received bins, not the exact bitmap.
        self.last_frame_missing_bins = tuple()
        self._captured_frame_count += 1
        return payload.copy(), frame_bfpexp

    def start(self) -> None:
        resolved_device = resolve_audio_device(self.cfg.device)
        self.cfg.device = resolved_device
        capture_mode = "raw"
        capture_extra_args: Optional[list[str]] = None
        resolved_backend, _base_cmd = resolve_capture_command(
            resolved_device,
            self.cfg.sample_rate,
            backend=self.cfg.capture_backend,
            capture_binary=self.cfg.capture_binary,
            capture_telemetry=self.cfg.capture_telemetry,
            capture_realign_initial_word_skip=self.cfg.capture_realign_initial_word_skip,
            capture_realign_swap_channels=self.cfg.capture_realign_swap_channels,
        )
        if self.cfg.use_i2s_tags and resolved_backend == CAPTURE_BACKEND_NATIVE:
            capture_mode = "fft-raw"
            capture_extra_args = [
                "--packet-index-shift", str(self.cfg.packet_index_shift),
                "--packet-index-bits", str(self.cfg.packet_index_bits),
                "--fft-packet-index-base", str(self.cfg.fft_packet_index_base),
                "--tag-shift", str(self.cfg.tag_shift),
                "--tag-mask", str(self.cfg.tag_mask),
                "--payload-bits", str(self.cfg.payload_bits),
                "--tag-idle", str(self.cfg.tag_idle),
                "--tag-bfpexp", str(self.cfg.tag_bfpexp),
                "--tag-fft", str(self.cfg.tag_fft),
                "--fft-frame-bins", str(self.cfg.frame_bins),
                "--bfpexp-hold-pairs", str(self.cfg.bfpexp_pairs_required),
                "--loss-tolerance-pairs", str(self.cfg.loss_tolerance_pairs),
            ]
            if not self.cfg.require_bfpexp_before_fft:
                capture_extra_args.append("--allow-fft-without-bfpexp")
        resolved_backend, cmd = resolve_capture_command(
            resolved_device,
            self.cfg.sample_rate,
            backend=self.cfg.capture_backend,
            capture_binary=self.cfg.capture_binary,
            capture_telemetry=self.cfg.capture_telemetry,
            capture_realign_initial_word_skip=self.cfg.capture_realign_initial_word_skip,
            capture_realign_swap_channels=self.cfg.capture_realign_swap_channels,
            capture_mode=capture_mode,
            capture_extra_args=capture_extra_args,
        )
        self._byte_buffer.clear()
        self._tagged_realigner.reset()
        self._tagged_pair_buffer = np.empty((0, 2), dtype=np.int32)
        self.last_frame_bfpexp = 0
        self.last_frame_had_explicit_bfpexp = False
        self.last_frame_missing_bins = tuple()
        self._captured_frame_count = 0
        self._helper_decodes_tagged_fft_frames = (
            resolved_backend == CAPTURE_BACKEND_NATIVE and capture_mode == "fft-raw"
        )
        self._helper_handles_tagged_alignment = (
            not self._helper_decodes_tagged_fft_frames
            and
            resolved_backend == CAPTURE_BACKEND_NATIVE
            and helper_tagged_realign_enabled(
                self.cfg.capture_realign_initial_word_skip,
                self.cfg.capture_realign_swap_channels,
            )
        )
        self.drain_capture_telemetry_events()
        try:
            self._proc = start_capture_process(
                resolved_device,
                self.cfg.sample_rate,
                backend=self.cfg.capture_backend,
                capture_binary=self.cfg.capture_binary,
                capture_telemetry=self.cfg.capture_telemetry,
                capture_realign_initial_word_skip=self.cfg.capture_realign_initial_word_skip,
                capture_realign_swap_channels=self.cfg.capture_realign_swap_channels,
                capture_stderr=self.cfg.capture_telemetry,
                capture_mode=capture_mode,
                capture_extra_args=capture_extra_args,
            )
            if self.cfg.capture_telemetry and self._proc.stderr is not None:
                self._stderr_thread = threading.Thread(target=self._capture_stderr_loop, daemon=True)
                self._stderr_thread.start()
            self._setup_gpio()
        except Exception:
            if self._proc is not None:
                stop_process(self._proc)
                self._proc = None
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1.0)
                self._stderr_thread = None
            self._teardown_gpio()
            raise
        print("Starting:", " ".join(cmd), flush=True)

    def stop(self) -> None:
        self._set_done(False)
        if self._proc is not None:
            stop_process(self._proc)
            self._proc = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None
        self._byte_buffer.clear()
        self._tagged_realigner.reset()
        self._tagged_pair_buffer = np.empty((0, 2), dtype=np.int32)
        self.last_frame_bfpexp = 0
        self.last_frame_had_explicit_bfpexp = False
        self.last_frame_missing_bins = tuple()
        self._captured_frame_count = 0
        self._helper_handles_tagged_alignment = False
        self._helper_decodes_tagged_fft_frames = False
        self._teardown_gpio()

    def read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        frame_bfpexp = 0
        if self._helper_decodes_tagged_fft_frames:
            tagged_frame = self._read_frame_from_helper_tagged_fft()
            if tagged_frame is None:
                return None
            pairs, frame_bfpexp = tagged_frame
        elif self.cfg.use_i2s_tags:
            tagged_frame = self._read_frame_from_i2s_tags()
            if tagged_frame is None:
                return None
            pairs, frame_bfpexp = tagged_frame
        else:
            if not self._wait_for_fft_window():
                return None

            pairs = self._pop_pairs(self.cfg.frame_bins, exact=True)
            if pairs is None:
                return None
            self.last_frame_missing_bins = tuple()

        if self.cfg.apply_bfpexp and frame_bfpexp != 0:
            real = np.ldexp(pairs[:, 0].astype(np.float32), frame_bfpexp)
            imag = np.ldexp(pairs[:, 1].astype(np.float32), frame_bfpexp)
        else:
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
