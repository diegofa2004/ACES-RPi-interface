import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np


AUTO_AUDIO_DEVICE = "auto"
BYTES_PER_STEREO_FRAME = 8
DEFAULT_CAPTURE_RATE_HZ = 48828
CAPTURE_BACKEND_AUTO = "auto"
CAPTURE_BACKEND_ARECORD = "arecord"
CAPTURE_BACKEND_NATIVE = "alsa-c"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw, 0)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_nonneg_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw, 0)
    except ValueError:
        return default
    return value if value >= 0 else default


# Keep ALSA periods short enough that each stdout write stays well below the
# typical Linux pipe capacity. Large period bursts can block arecord on stdout
# long enough to starve ALSA and trigger an overrun in the analyzer.
DEFAULT_ARECORD_BUFFER_TIME_US = _env_int("FPGAFFT_ARECORD_BUFFER_TIME_US", 1000000)
DEFAULT_ARECORD_PERIOD_TIME_US = _env_int("FPGAFFT_ARECORD_PERIOD_TIME_US", 20000)
DEFAULT_ARECORD_PIPE_SIZE_BYTES = _env_int("FPGAFFT_ARECORD_PIPE_SIZE_BYTES", 1 << 20)
DEFAULT_CAPTURE_BACKEND = (os.environ.get("FPGAFFT_CAPTURE_BACKEND") or CAPTURE_BACKEND_AUTO).strip().lower()
DEFAULT_NATIVE_CAPTURE_READ_FRAMES = _env_int("FPGAFFT_NATIVE_CAPTURE_READ_FRAMES", 2048)
DEFAULT_NATIVE_CAPTURE_PERIOD_FRAMES = _env_int("FPGAFFT_NATIVE_CAPTURE_PERIOD_FRAMES", 512)
DEFAULT_NATIVE_CAPTURE_BUFFER_FRAMES = _env_int("FPGAFFT_NATIVE_CAPTURE_BUFFER_FRAMES", 8192)
DEFAULT_NATIVE_CAPTURE_QUEUE_CHUNKS = _env_int("FPGAFFT_NATIVE_CAPTURE_QUEUE_CHUNKS", 512)
DEFAULT_NATIVE_CAPTURE_PIPE_SIZE_BYTES = _env_int("FPGAFFT_NATIVE_CAPTURE_PIPE_SIZE_BYTES", 1 << 20)
DEFAULT_NATIVE_CAPTURE_STATS_INTERVAL_MS = _env_nonneg_int("FPGAFFT_NATIVE_CAPTURE_STATS_INTERVAL_MS", 0)
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
_DEFAULT_ALIGNMENT_SEARCH_PAIR_LIMIT = 512
_DEFAULT_ALIGNMENT_CONFIRM_PAIRS = 64
_DEFAULT_ALIGNMENT_VALIDATE_PAIRS = 64
_DEFAULT_ALIGNMENT_LOCK_MIN_SCORE = 240
_DEFAULT_ALIGNMENT_LOCK_MIN_MARGIN = 140
_DEFAULT_ALIGNMENT_MIN_GOOD_RATIO = 0.80
_DEFAULT_ALIGNMENT_MIN_RESERVED_RATIO = 0.95
_DEFAULT_ALIGNMENT_MAX_RAW_WORDS = 4096


@dataclass(frozen=True)
class TaggedI2SAlignment:
    bit_offset: int
    initial_word_skip: int
    swap_channels: bool
    score: int


@dataclass(frozen=True)
class TaggedI2SAlignmentMetrics:
    pair_count: int
    good_pairs: int
    reserved_zero_good_pairs: int
    active_pairs: int
    fft_pairs: int
    bfpexp_pairs: int
    bfpexp_equal_pairs: int
    invalid_pairs: int
    longest_fft_run: int
    bfpexp_to_fft_transitions: int


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


def _collect_tagged_alignment_metrics(
    words_u32: np.ndarray,
    *,
    tag_shift: int,
    tag_mask: int,
    payload_bits: int,
    tag_idle: int,
    tag_bfpexp: int,
    tag_fft: int,
    search_pair_limit: int,
) -> TaggedI2SAlignmentMetrics:
    if words_u32.size < 2:
        return TaggedI2SAlignmentMetrics(
            pair_count=0,
            good_pairs=0,
            reserved_zero_good_pairs=0,
            active_pairs=0,
            fft_pairs=0,
            bfpexp_pairs=0,
            bfpexp_equal_pairs=0,
            invalid_pairs=0,
            longest_fft_run=0,
            bfpexp_to_fft_transitions=0,
        )

    pair_count = min(words_u32.size // 2, search_pair_limit)

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

    longest_fft_run = 0
    current_fft_run = 0
    bfpexp_to_fft_transitions = 0
    prev_kind = "invalid"
    for pair_index in range(pair_count):
        if not bool(good_pairs[pair_index]):
            kind = "invalid"
            current_fft_run = 0
        elif bool(bfpexp_pairs[pair_index]):
            kind = "bfpexp"
            current_fft_run = 0
        elif bool(fft_pairs[pair_index]):
            kind = "fft"
            current_fft_run += 1
            if current_fft_run > longest_fft_run:
                longest_fft_run = current_fft_run
        else:
            kind = "idle"
            current_fft_run = 0

        if prev_kind == "bfpexp" and kind == "fft":
            bfpexp_to_fft_transitions += 1
        prev_kind = kind

    return TaggedI2SAlignmentMetrics(
        pair_count=pair_count,
        good_pairs=int(good_pairs.sum()),
        reserved_zero_good_pairs=int(reserved_zero[good_pairs].sum()),
        active_pairs=int(active_pairs.sum()),
        fft_pairs=int(fft_pairs.sum()),
        bfpexp_pairs=int(bfpexp_pairs.sum()),
        bfpexp_equal_pairs=int(bfpexp_equal.sum()),
        invalid_pairs=int((~good_pairs).sum()),
        longest_fft_run=longest_fft_run,
        bfpexp_to_fft_transitions=bfpexp_to_fft_transitions,
    )


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
    metrics = _collect_tagged_alignment_metrics(
        words_u32,
        tag_shift=tag_shift,
        tag_mask=tag_mask,
        payload_bits=payload_bits,
        tag_idle=tag_idle,
        tag_bfpexp=tag_bfpexp,
        tag_fft=tag_fft,
        search_pair_limit=search_pair_limit,
    )
    if metrics.pair_count < 4:
        return -1_000_000

    score = 0
    score += metrics.good_pairs * 120
    score += metrics.reserved_zero_good_pairs * 80
    score += metrics.active_pairs * 35
    score += metrics.fft_pairs * 18
    score += metrics.bfpexp_equal_pairs * 110
    score += metrics.longest_fft_run * 25
    score += metrics.bfpexp_to_fft_transitions * 160
    score -= metrics.invalid_pairs * 260
    score -= max(0, metrics.good_pairs - metrics.reserved_zero_good_pairs) * 180

    if metrics.active_pairs == 0:
        score -= 2_000
    if metrics.longest_fft_run < 4:
        score -= 600

    return score


def _rank_tagged_alignment_candidates(
    stereo: np.ndarray,
    *,
    tag_shift: int,
    tag_mask: int,
    payload_bits: int,
    tag_idle: int,
    tag_bfpexp: int,
    tag_fft: int,
    search_pair_limit: int,
) -> list[TaggedI2SAlignment]:
    stereo_i32 = np.asarray(stereo, dtype=np.int32)
    if stereo_i32.size < 8 or (stereo_i32.size % 2) != 0:
        return []

    words_u32 = stereo_i32.reshape(-1).astype(np.uint32, copy=False)
    candidates: list[TaggedI2SAlignment] = []

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

                candidates.append(
                    TaggedI2SAlignment(
                        bit_offset=bit_offset,
                        initial_word_skip=initial_word_skip,
                        swap_channels=swap_channels,
                        score=score,
                    )
                )

    candidates.sort(key=lambda alignment: alignment.score, reverse=True)
    return candidates


def _detect_best_alignment_candidates(
    stereo: np.ndarray,
    *,
    tag_shift: int,
    tag_mask: int,
    payload_bits: int,
    tag_idle: int,
    tag_bfpexp: int,
    tag_fft: int,
    search_pair_limit: int,
) -> tuple[Optional[TaggedI2SAlignment], Optional[TaggedI2SAlignment]]:
    candidates = _rank_tagged_alignment_candidates(
        stereo,
        tag_shift=tag_shift,
        tag_mask=tag_mask,
        payload_bits=payload_bits,
        tag_idle=tag_idle,
        tag_bfpexp=tag_bfpexp,
        tag_fft=tag_fft,
        search_pair_limit=search_pair_limit,
    )
    if not candidates:
        return None, None
    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    return best, runner_up


def detect_tagged_i2s_alignment(
    stereo: np.ndarray,
    *,
    tag_shift: int = _DEFAULT_TAG_SHIFT,
    tag_mask: int = _DEFAULT_TAG_MASK,
    payload_bits: int = _DEFAULT_PAYLOAD_BITS,
    tag_idle: int = _DEFAULT_TAG_IDLE,
    tag_bfpexp: int = _DEFAULT_TAG_BFPEXP,
    tag_fft: int = _DEFAULT_TAG_FFT,
    search_pair_limit: int = _DEFAULT_ALIGNMENT_SEARCH_PAIR_LIMIT,
) -> Optional[TaggedI2SAlignment]:
    best, _runner_up = _detect_best_alignment_candidates(
        stereo,
        tag_shift=tag_shift,
        tag_mask=tag_mask,
        payload_bits=payload_bits,
        tag_idle=tag_idle,
        tag_bfpexp=tag_bfpexp,
        tag_fft=tag_fft,
        search_pair_limit=search_pair_limit,
    )
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
        search_pair_limit: int = _DEFAULT_ALIGNMENT_SEARCH_PAIR_LIMIT,
        confirm_pairs: int = _DEFAULT_ALIGNMENT_CONFIRM_PAIRS,
        validate_pairs: int = _DEFAULT_ALIGNMENT_VALIDATE_PAIRS,
        min_lock_score: int = _DEFAULT_ALIGNMENT_LOCK_MIN_SCORE,
        min_lock_margin: int = _DEFAULT_ALIGNMENT_LOCK_MIN_MARGIN,
        min_good_ratio: float = _DEFAULT_ALIGNMENT_MIN_GOOD_RATIO,
        min_reserved_ratio: float = _DEFAULT_ALIGNMENT_MIN_RESERVED_RATIO,
        max_raw_words: int = _DEFAULT_ALIGNMENT_MAX_RAW_WORDS,
        preferred_swap_channels: Optional[bool] = None,
    ):
        self._tag_shift = tag_shift
        self._tag_mask = tag_mask
        self._payload_bits = payload_bits
        self._tag_idle = tag_idle
        self._tag_bfpexp = tag_bfpexp
        self._tag_fft = tag_fft
        self._search_pair_limit = max(8, int(search_pair_limit))
        self._confirm_pairs = max(1, int(confirm_pairs))
        self._validate_pairs = max(4, int(validate_pairs))
        self._min_lock_score = int(min_lock_score)
        self._min_lock_margin = int(min_lock_margin)
        self._min_good_ratio = float(min_good_ratio)
        self._min_reserved_ratio = float(min_reserved_ratio)
        self._max_raw_words = max(32, int(max_raw_words))
        self._preferred_swap_channels = preferred_swap_channels
        self._alignment: Optional[TaggedI2SAlignment] = None
        self._raw_word_buffer = np.empty(0, dtype=np.uint32)
        self._aligned_word_buffer = np.empty(0, dtype=np.uint32)
        self._initial_skip_done = False
        self._output_enabled = False

    @property
    def alignment(self) -> Optional[TaggedI2SAlignment]:
        return self._alignment

    def reset(self) -> None:
        self._alignment = None
        self._raw_word_buffer = np.empty(0, dtype=np.uint32)
        self._aligned_word_buffer = np.empty(0, dtype=np.uint32)
        self._initial_skip_done = False
        self._output_enabled = False

    def _trim_raw_word_buffer(self) -> None:
        if self._raw_word_buffer.size <= self._max_raw_words:
            return
        self._raw_word_buffer = self._raw_word_buffer[-self._max_raw_words :].copy()

    def _try_lock_alignment(self) -> bool:
        search_word_count = (self._raw_word_buffer.size // 2) * 2
        if search_word_count < 8:
            return False
        search_words = self._raw_word_buffer[:search_word_count]

        candidates = _rank_tagged_alignment_candidates(
            search_words.view(np.int32).reshape(-1, 2),
            tag_shift=self._tag_shift,
            tag_mask=self._tag_mask,
            payload_bits=self._payload_bits,
            tag_idle=self._tag_idle,
            tag_bfpexp=self._tag_bfpexp,
            tag_fft=self._tag_fft,
            search_pair_limit=self._search_pair_limit,
        )
        if not candidates:
            return False
        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None

        if self._preferred_swap_channels is not None:
            for candidate in candidates:
                if candidate.score != best.score:
                    break
                if candidate.bit_offset != best.bit_offset or candidate.initial_word_skip != best.initial_word_skip:
                    continue
                if candidate.swap_channels == self._preferred_swap_channels:
                    best = candidate
                    break
        if best is None or best.score < self._min_lock_score:
            return False

        if runner_up is not None and search_word_count >= 32:
            same_boundary = (
                best.bit_offset == runner_up.bit_offset
                and best.initial_word_skip == runner_up.initial_word_skip
            )
            if ((best.score - runner_up.score) < self._min_lock_margin) and (not same_boundary):
                return False

        self._alignment = best
        if self._preferred_swap_channels is None:
            self._preferred_swap_channels = best.swap_channels
        self._initial_skip_done = False
        self._output_enabled = False
        return True

    def _reset_for_relock(self, raw_snapshot: Optional[np.ndarray] = None, *, drop_words: int = 1) -> None:
        preserved = np.empty(0, dtype=np.uint32)
        if raw_snapshot is not None and raw_snapshot.size > drop_words:
            preserved = raw_snapshot[drop_words:]
            if preserved.size > self._max_raw_words:
                preserved = preserved[-self._max_raw_words :]
            preserved = preserved.copy()

        self._alignment = None
        self._raw_word_buffer = preserved
        self._aligned_word_buffer = np.empty(0, dtype=np.uint32)
        self._initial_skip_done = False
        self._output_enabled = False

    def _consume_aligned_words(self) -> tuple[np.ndarray, np.ndarray]:
        raw_snapshot = self._raw_word_buffer.copy()
        if self._alignment is None:
            return raw_snapshot, np.empty(0, dtype=np.uint32)

        if self._alignment.bit_offset == 0:
            new_aligned = raw_snapshot
            self._raw_word_buffer = np.empty(0, dtype=np.uint32)
            return raw_snapshot, new_aligned

        if raw_snapshot.size < 2:
            return raw_snapshot, np.empty(0, dtype=np.uint32)

        new_aligned = _reframe_tagged_words(raw_snapshot, self._alignment.bit_offset)
        self._raw_word_buffer = raw_snapshot[-1:].copy()
        return raw_snapshot, new_aligned

    def _window_is_valid(self, pair_words: np.ndarray) -> bool:
        if pair_words.size == 0:
            return True

        window = np.asarray(pair_words, dtype=np.int32)
        if window.ndim != 2 or window.shape[1] != 2:
            window = window.reshape(-1, 2)
        if window.shape[0] < 2:
            return True

        window = window[-min(window.shape[0], self._validate_pairs) :]
        metrics = _collect_tagged_alignment_metrics(
            window.reshape(-1).astype(np.uint32, copy=False),
            tag_shift=self._tag_shift,
            tag_mask=self._tag_mask,
            payload_bits=self._payload_bits,
            tag_idle=self._tag_idle,
            tag_bfpexp=self._tag_bfpexp,
            tag_fft=self._tag_fft,
            search_pair_limit=window.shape[0],
        )
        if metrics.pair_count == 0:
            return False

        good_ratio = metrics.good_pairs / metrics.pair_count
        reserved_ratio = (
            metrics.reserved_zero_good_pairs / metrics.good_pairs if metrics.good_pairs > 0 else 0.0
        )
        return good_ratio >= self._min_good_ratio and reserved_ratio >= self._min_reserved_ratio

    def _pair_is_valid(self, pair_words: np.ndarray) -> bool:
        pair = np.asarray(pair_words, dtype=np.int32).reshape(-1, 2)
        if pair.shape[0] != 1:
            raise ValueError("_pair_is_valid expects exactly one stereo pair")

        metrics = _collect_tagged_alignment_metrics(
            pair.reshape(-1).astype(np.uint32, copy=False),
            tag_shift=self._tag_shift,
            tag_mask=self._tag_mask,
            payload_bits=self._payload_bits,
            tag_idle=self._tag_idle,
            tag_bfpexp=self._tag_bfpexp,
            tag_fft=self._tag_fft,
            search_pair_limit=1,
        )
        return metrics.pair_count == 1 and metrics.good_pairs == 1 and metrics.reserved_zero_good_pairs == 1

    def _drop_leading_invalid_pairs(self, candidate_words: np.ndarray) -> np.ndarray:
        pair_word_count = (candidate_words.size // 2) * 2
        if pair_word_count == 0:
            return candidate_words

        pair_words = candidate_words[:pair_word_count].reshape(-1, 2)
        drop_pairs = 0
        while drop_pairs < pair_words.shape[0]:
            if self._pair_is_valid(pair_words[drop_pairs : drop_pairs + 1]):
                break
            drop_pairs += 1

        if drop_pairs == 0:
            return candidate_words
        return candidate_words[drop_pairs * 2 :]

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
        self._trim_raw_word_buffer()

        for _attempt in range(3):
            if self._alignment is None and (not self._try_lock_alignment()):
                return np.empty((0, 2), dtype=np.int32)

            raw_snapshot, new_aligned = self._consume_aligned_words()
            if new_aligned.size == 0:
                return np.empty((0, 2), dtype=np.int32)

            if self._aligned_word_buffer.size == 0:
                candidate_words = new_aligned
            else:
                candidate_words = np.concatenate((self._aligned_word_buffer, new_aligned))

            if (not self._initial_skip_done) and self._alignment is not None and self._alignment.initial_word_skip:
                if candidate_words.size <= self._alignment.initial_word_skip:
                    self._aligned_word_buffer = candidate_words
                    return np.empty((0, 2), dtype=np.int32)
                candidate_words = candidate_words[self._alignment.initial_word_skip :]
                self._initial_skip_done = True
            else:
                self._initial_skip_done = True

            candidate_words = self._drop_leading_invalid_pairs(candidate_words)

            pair_word_count = (candidate_words.size // 2) * 2
            if pair_word_count == 0:
                self._aligned_word_buffer = candidate_words
                return np.empty((0, 2), dtype=np.int32)

            pair_words = candidate_words[:pair_word_count].reshape(-1, 2)
            swap_output = False
            if self._preferred_swap_channels is None:
                swap_output = bool(self._alignment is not None and self._alignment.swap_channels)
            else:
                swap_output = self._preferred_swap_channels
            if swap_output:
                pair_words = pair_words[:, ::-1]

            if not self._window_is_valid(pair_words):
                self._reset_for_relock(raw_snapshot, drop_words=1)
                continue

            if (not self._output_enabled) and pair_words.shape[0] < self._confirm_pairs:
                self._aligned_word_buffer = candidate_words
                return np.empty((0, 2), dtype=np.int32)

            self._output_enabled = True
            self._aligned_word_buffer = candidate_words[pair_word_count:]
            return pair_words.astype(np.uint32, copy=False).view(np.int32)

        self._reset_for_relock()
        return np.empty((0, 2), dtype=np.int32)


def build_arecord_cmd(device: str, rate: int) -> list[str]:
    period_time_us = max(1000, DEFAULT_ARECORD_PERIOD_TIME_US)
    buffer_time_us = max(period_time_us * 4, DEFAULT_ARECORD_BUFFER_TIME_US)
    return [
        "arecord",
        "-q",
        "-D",
        device,
        "-B",
        str(buffer_time_us),
        "-F",
        str(period_time_us),
        "-f",
        "S32_LE",
        "-c",
        "2",
        "-r",
        str(rate),
        "-t",
        "raw",
    ]


def find_native_capture_binary() -> Optional[str]:
    env_path = os.environ.get("FPGAFFT_CAPTURE_BINARY", "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    module_dir = Path(__file__).resolve().parent
    for name in ("alsa_logger", "alsa_capture"):
        candidate = module_dir / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


def _normalize_capture_backend_name(backend: Optional[str]) -> str:
    normalized = (backend or DEFAULT_CAPTURE_BACKEND or CAPTURE_BACKEND_AUTO).strip().lower()
    if normalized in ("", CAPTURE_BACKEND_AUTO):
        return CAPTURE_BACKEND_AUTO
    if normalized in (CAPTURE_BACKEND_ARECORD,):
        return CAPTURE_BACKEND_ARECORD
    if normalized in (CAPTURE_BACKEND_NATIVE, "alsa", "native", "c"):
        return CAPTURE_BACKEND_NATIVE
    raise ValueError(f"Unknown capture backend: {backend}")


def resolve_capture_command(
    device: str,
    rate: int,
    *,
    backend: Optional[str] = None,
    capture_binary: Optional[str] = None,
) -> tuple[str, list[str]]:
    normalized_backend = _normalize_capture_backend_name(backend)

    if normalized_backend == CAPTURE_BACKEND_ARECORD:
        return CAPTURE_BACKEND_ARECORD, build_arecord_cmd(device, rate)

    binary_path = capture_binary or find_native_capture_binary()
    if normalized_backend == CAPTURE_BACKEND_NATIVE:
        if not binary_path:
            raise RuntimeError(
                "Native C capture backend requested but no compiled helper was found. "
                "Build rpi3b_i2s_fft/alsa_logger first or use --capture-backend arecord."
            )
        return CAPTURE_BACKEND_NATIVE, build_native_capture_cmd(binary_path, device, rate)

    if binary_path:
        return CAPTURE_BACKEND_NATIVE, build_native_capture_cmd(binary_path, device, rate)

    return CAPTURE_BACKEND_ARECORD, build_arecord_cmd(device, rate)


def build_native_capture_cmd(binary_path: str, device: str, rate: int) -> list[str]:
    cmd = [
        binary_path,
        "--device",
        device,
        "--rate",
        str(rate),
        "--mode",
        "raw",
        "--read-frames",
        str(DEFAULT_NATIVE_CAPTURE_READ_FRAMES),
        "--period-frames",
        str(DEFAULT_NATIVE_CAPTURE_PERIOD_FRAMES),
        "--buffer-frames",
        str(DEFAULT_NATIVE_CAPTURE_BUFFER_FRAMES),
        "--queue-chunks",
        str(DEFAULT_NATIVE_CAPTURE_QUEUE_CHUNKS),
        "--stats-interval-ms",
        str(DEFAULT_NATIVE_CAPTURE_STATS_INTERVAL_MS),
    ]
    if DEFAULT_NATIVE_CAPTURE_PIPE_SIZE_BYTES > 0:
        cmd.extend(["--pipe-size-bytes", str(DEFAULT_NATIVE_CAPTURE_PIPE_SIZE_BYTES)])
    return cmd


def build_capture_cmd(
    device: str,
    rate: int,
    *,
    backend: Optional[str] = None,
    capture_binary: Optional[str] = None,
) -> list[str]:
    _resolved_backend, cmd = resolve_capture_command(
        device,
        rate,
        backend=backend,
        capture_binary=capture_binary,
    )
    return cmd


def start_arecord_process(device: str, rate: int) -> subprocess.Popen:
    cmd = build_arecord_cmd(device, rate)
    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": None,
        "bufsize": 0,
    }
    try:
        if DEFAULT_ARECORD_PIPE_SIZE_BYTES > 0:
            try:
                return subprocess.Popen(
                    cmd,
                    pipesize=DEFAULT_ARECORD_PIPE_SIZE_BYTES,
                    **popen_kwargs,
                )
            except (TypeError, ValueError, PermissionError):
                pass

        return subprocess.Popen(cmd, **popen_kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'arecord' command was not found. Install alsa-utils on the Raspberry Pi."
        ) from exc


def start_capture_process(
    device: str,
    rate: int,
    *,
    backend: Optional[str] = None,
    capture_binary: Optional[str] = None,
) -> subprocess.Popen:
    resolved_backend, cmd = resolve_capture_command(
        device,
        rate,
        backend=backend,
        capture_binary=capture_binary,
    )
    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": None,
        "bufsize": 0,
    }
    try:
        pipe_size_bytes = (
            DEFAULT_NATIVE_CAPTURE_PIPE_SIZE_BYTES
            if resolved_backend == CAPTURE_BACKEND_NATIVE
            else DEFAULT_ARECORD_PIPE_SIZE_BYTES
        )
        if pipe_size_bytes > 0:
            try:
                return subprocess.Popen(
                    cmd,
                    pipesize=pipe_size_bytes,
                    **popen_kwargs,
                )
            except (TypeError, ValueError, PermissionError):
                pass

        return subprocess.Popen(cmd, **popen_kwargs)
    except FileNotFoundError as exc:
        if resolved_backend == CAPTURE_BACKEND_NATIVE:
            raise RuntimeError(
                f"Native capture helper not found: {cmd[0]}. "
                "Build rpi3b_i2s_fft/alsa_logger or use --capture-backend arecord."
            ) from exc
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
