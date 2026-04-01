import os
import re
import subprocess
from dataclasses import dataclass
from typing import BinaryIO, Optional

import numpy as np


AUTO_AUDIO_DEVICE = "auto"
BYTES_PER_STEREO_FRAME = 8
DEFAULT_CAPTURE_RATE_HZ = 48828
DEFAULT_ARECORD_BUFFER_TIME_US = 250000
DEFAULT_ARECORD_PERIOD_TIME_US = 50000
_CAPTURE_DEVICE_RE = re.compile(r"^card\s+(?P<card>\d+):.*device\s+(?P<device>\d+):", re.IGNORECASE)
_PREFERRED_CAPTURE_KEYWORDS = (
    "aces-fpgafft",
    "fpgafft",
    "googlevoicehat",
    "voicehat",
    "voice hat",
    "aiy",
    "i2s",
    "snd_rpi",
    "sndrpi",
)

_DEFAULT_TAG_SHIFT = 30
_DEFAULT_TAG_MASK = 0x3
_DEFAULT_PAYLOAD_BITS = 18
_DEFAULT_TAG_IDLE = 0
_DEFAULT_TAG_BFPEXP = 1
_DEFAULT_TAG_FFT = 2


@dataclass(frozen=True)
class TaggedI2SAlignment:
    bit_offset: int
    initial_word_skip: int
    swap_channels: bool
    score: int


def _reframe_tagged_words(words_u32: np.ndarray, bit_offset: int) -> np.ndarray:
    if words_u32.size == 0:
        return np.empty(0, dtype=np.uint32)
    if bit_offset == 0:
        return words_u32.astype(np.uint32, copy=True)
    if words_u32.size < 2:
        return np.empty(0, dtype=np.uint32)

    left = (words_u32[:-1].astype(np.uint64) << bit_offset) & 0xFFFFFFFF
    right = words_u32[1:].astype(np.uint64) >> np.uint64(32 - bit_offset)
    return (left | right).astype(np.uint32)


def _score_tagged_alignment_candidate(
    words_u32: np.ndarray,
    *,
    tag_shift: int,
    tag_mask: int,
    payload_bits: int,
    tag_idle: int,
    tag_bfpexp: int,
    tag_fft: int,
    search_pair_limit: int,
) -> int:
    if words_u32.size < 8:
        return -1_000_000

    pair_count = min(words_u32.size // 2, search_pair_limit)
    if pair_count < 4:
        return -1_000_000

    pairs = words_u32[: pair_count * 2].reshape(-1, 2)
    left = pairs[:, 0]
    right = pairs[:, 1]

    tags_l = (left >> np.uint32(tag_shift)) & np.uint32(tag_mask)
    tags_r = (right >> np.uint32(tag_shift)) & np.uint32(tag_mask)
    known_l = np.isin(tags_l, (tag_idle, tag_bfpexp, tag_fft))
    known_r = np.isin(tags_r, (tag_idle, tag_bfpexp, tag_fft))
    tag_match = tags_l == tags_r
    good_pairs = known_l & known_r & tag_match

    reserved_width = max(0, tag_shift - payload_bits)
    reserved_mask = (
        np.uint32(((1 << reserved_width) - 1) << payload_bits) if reserved_width > 0 else np.uint32(0)
    )
    if reserved_width > 0:
        reserved_zero = (((left | right) & reserved_mask) == 0)
    else:
        reserved_zero = np.ones(pair_count, dtype=bool)

    bfpexp_pairs = good_pairs & (tags_l == tag_bfpexp)
    fft_pairs = good_pairs & (tags_l == tag_fft)
    active_pairs = bfpexp_pairs | fft_pairs
    bfpexp_equal = bfpexp_pairs & (left == right)

    score = 0
    score += int(good_pairs.sum()) * 100
    score += int(reserved_zero[good_pairs].sum()) * 60
    score += int(active_pairs.sum()) * 30
    score += int(fft_pairs.sum()) * 15
    score += int(bfpexp_equal.sum()) * 90
    score -= int((~good_pairs).sum()) * 220
    score -= int((good_pairs & (~reserved_zero)).sum()) * 140

    if active_pairs.sum() == 0:
        score -= 2_000
    if reserved_zero[good_pairs].sum() < max(2, good_pairs.sum() // 2):
        score -= 800

    return score


def detect_tagged_i2s_alignment(
    stereo: np.ndarray,
    *,
    tag_shift: int = _DEFAULT_TAG_SHIFT,
    tag_mask: int = _DEFAULT_TAG_MASK,
    payload_bits: int = _DEFAULT_PAYLOAD_BITS,
    tag_idle: int = _DEFAULT_TAG_IDLE,
    tag_bfpexp: int = _DEFAULT_TAG_BFPEXP,
    tag_fft: int = _DEFAULT_TAG_FFT,
    search_pair_limit: int = 128,
) -> Optional[TaggedI2SAlignment]:
    stereo_i32 = np.asarray(stereo, dtype=np.int32)
    if stereo_i32.size < 8 or (stereo_i32.size % 2) != 0:
        return None

    words_u32 = stereo_i32.reshape(-1).astype(np.uint32, copy=False)
    best: Optional[TaggedI2SAlignment] = None

    for bit_offset in range(32):
        reframed = _reframe_tagged_words(words_u32, bit_offset)
        if reframed.size < 8:
            continue
        for initial_word_skip in (0, 1):
            if reframed.size <= initial_word_skip + 3:
                continue
            usable = reframed[initial_word_skip:]
            for swap_channels in (False, True):
                candidate = usable
                if swap_channels:
                    candidate = candidate[: (candidate.size // 2) * 2].reshape(-1, 2)[:, ::-1].reshape(-1)

                score = _score_tagged_alignment_candidate(
                    candidate,
                    tag_shift=tag_shift,
                    tag_mask=tag_mask,
                    payload_bits=payload_bits,
                    tag_idle=tag_idle,
                    tag_bfpexp=tag_bfpexp,
                    tag_fft=tag_fft,
                    search_pair_limit=search_pair_limit,
                )

                alignment = TaggedI2SAlignment(
                    bit_offset=bit_offset,
                    initial_word_skip=initial_word_skip,
                    swap_channels=swap_channels,
                    score=score,
                )
                if best is None or alignment.score > best.score:
                    best = alignment

    if best is None or best.score < 120:
        return None

    return best


class TaggedI2SRealigner:
    def __init__(
        self,
        *,
        tag_shift: int = _DEFAULT_TAG_SHIFT,
        tag_mask: int = _DEFAULT_TAG_MASK,
        payload_bits: int = _DEFAULT_PAYLOAD_BITS,
        tag_idle: int = _DEFAULT_TAG_IDLE,
        tag_bfpexp: int = _DEFAULT_TAG_BFPEXP,
        tag_fft: int = _DEFAULT_TAG_FFT,
    ):
        self._tag_shift = tag_shift
        self._tag_mask = tag_mask
        self._payload_bits = payload_bits
        self._tag_idle = tag_idle
        self._tag_bfpexp = tag_bfpexp
        self._tag_fft = tag_fft
        self._alignment: Optional[TaggedI2SAlignment] = None
        self._raw_word_buffer = np.empty(0, dtype=np.uint32)
        self._aligned_word_buffer = np.empty(0, dtype=np.uint32)
        self._initial_skip_done = False

    @property
    def alignment(self) -> Optional[TaggedI2SAlignment]:
        return self._alignment

    def reset(self) -> None:
        self._alignment = None
        self._raw_word_buffer = np.empty(0, dtype=np.uint32)
        self._aligned_word_buffer = np.empty(0, dtype=np.uint32)
        self._initial_skip_done = False

    def push_pairs(self, stereo: np.ndarray) -> np.ndarray:
        stereo_i32 = np.asarray(stereo, dtype=np.int32)
        if stereo_i32.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        if stereo_i32.ndim != 2 or stereo_i32.shape[1] != 2:
            stereo_i32 = stereo_i32.reshape(-1, 2)

        words_u32 = stereo_i32.reshape(-1).astype(np.uint32, copy=False)
        if self._raw_word_buffer.size == 0:
            self._raw_word_buffer = words_u32.copy()
        else:
            self._raw_word_buffer = np.concatenate((self._raw_word_buffer, words_u32))

        if self._alignment is None:
            self._alignment = detect_tagged_i2s_alignment(
                self._raw_word_buffer.view(np.int32).reshape(-1, 2),
                tag_shift=self._tag_shift,
                tag_mask=self._tag_mask,
                payload_bits=self._payload_bits,
                tag_idle=self._tag_idle,
                tag_bfpexp=self._tag_bfpexp,
                tag_fft=self._tag_fft,
            )
            if self._alignment is None:
                return np.empty((0, 2), dtype=np.int32)

        if self._alignment.bit_offset == 0:
            new_aligned = self._raw_word_buffer
            self._raw_word_buffer = np.empty(0, dtype=np.uint32)
        else:
            if self._raw_word_buffer.size < 2:
                return np.empty((0, 2), dtype=np.int32)
            new_aligned = _reframe_tagged_words(self._raw_word_buffer, self._alignment.bit_offset)
            self._raw_word_buffer = self._raw_word_buffer[-1:].copy()

        if self._aligned_word_buffer.size == 0:
            self._aligned_word_buffer = new_aligned
        else:
            self._aligned_word_buffer = np.concatenate((self._aligned_word_buffer, new_aligned))

        if (not self._initial_skip_done) and self._alignment.initial_word_skip:
            if self._aligned_word_buffer.size <= self._alignment.initial_word_skip:
                return np.empty((0, 2), dtype=np.int32)
            self._aligned_word_buffer = self._aligned_word_buffer[self._alignment.initial_word_skip :]
            self._initial_skip_done = True
        else:
            self._initial_skip_done = True

        pair_word_count = (self._aligned_word_buffer.size // 2) * 2
        if pair_word_count == 0:
            return np.empty((0, 2), dtype=np.int32)

        pair_words = self._aligned_word_buffer[:pair_word_count].reshape(-1, 2)
        self._aligned_word_buffer = self._aligned_word_buffer[pair_word_count:]

        if self._alignment.swap_channels:
            pair_words = pair_words[:, ::-1]

        return pair_words.astype(np.uint32, copy=False).view(np.int32)


def build_arecord_cmd(device: str, rate: int) -> list[str]:
    return [
        "arecord",
        "-q",
        "-D",
        device,
        "-B",
        str(DEFAULT_ARECORD_BUFFER_TIME_US),
        "-F",
        str(DEFAULT_ARECORD_PERIOD_TIME_US),
        "-f",
        "S32_LE",
        "-c",
        "2",
        "-r",
        str(rate),
        "-t",
        "raw",
    ]


def start_arecord_process(device: str, rate: int) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            build_arecord_cmd(device, rate),
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'arecord' command was not found. Install alsa-utils on the Raspberry Pi."
        ) from exc


def list_capture_devices() -> list[tuple[str, str]]:
    try:
        proc = subprocess.run(
            ["arecord", "-l"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'arecord' command was not found. Install alsa-utils on the Raspberry Pi."
        ) from exc

    devices: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        match = _CAPTURE_DEVICE_RE.match(line.strip())
        if not match:
            continue
        hw_name = f"hw:{match.group('card')},{match.group('device')}"
        devices.append((hw_name, line.strip()))
    return devices


def resolve_audio_device(device: str) -> str:
    requested = (device or "").strip()
    if requested and requested.lower() != AUTO_AUDIO_DEVICE:
        return requested

    env_device = os.environ.get("AUDIO_DEVICE", "").strip()
    if env_device:
        return env_device

    devices = list_capture_devices()
    if not devices:
        raise RuntimeError(
            "No ALSA capture device was found. "
            "Check the fpgafft overlay installation, reboot the Pi, and run 'arecord -l'."
        )

    if len(devices) == 1:
        return devices[0][0]

    for keyword in _PREFERRED_CAPTURE_KEYWORDS:
        for hw_name, description in devices:
            if keyword in description.lower():
                return hw_name

    found = ", ".join(f"{hw_name} ({description})" for hw_name, description in devices[:4])
    raise RuntimeError(
        "Multiple ALSA capture devices were found. "
        f"Run 'arecord -l' and pass '--device hw:X,Y'. Detected: {found}"
    )


def read_exactly(stream: BinaryIO, byte_count: int) -> bytes:
    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")

    chunks = bytearray()
    while len(chunks) < byte_count:
        piece = stream.read(byte_count - len(chunks))
        if not piece:
            break
        chunks.extend(piece)
    return bytes(chunks)


def trim_incomplete_frames(raw: bytes, bytes_per_frame: int = BYTES_PER_STEREO_FRAME) -> bytes:
    if bytes_per_frame <= 0:
        raise ValueError("bytes_per_frame must be positive")
    valid_size = len(raw) - (len(raw) % bytes_per_frame)
    return raw[:valid_size]


def stop_process(proc: subprocess.Popen, timeout: float = 2.0) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
