import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


EPSILON = 1e-6
DEFAULT_BAND_COUNT = 32
DEFAULT_USEFUL_BINS = 256

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENTO_FILENAME = os.path.join(MODULE_DIR, "evento.npy")
FFT_FILENAME = os.path.join(MODULE_DIR, "fft.npy")
SIMILARITY_STATE_FILENAME = os.path.join(MODULE_DIR, "similaridade.flag")
SIMILARITY_STATE_TMP_FILENAME = os.path.join(MODULE_DIR, "similaridade_tmp.flag")


@dataclass(frozen=True)
class DirectComparatorConfig:
    cooldown_seconds: float = 15.0
    poll_interval_seconds: float = 0.05
    min_consecutive_hits: int = 2
    score_history_size: int = 60
    absolute_threshold: float = 0.78
    margin_threshold: float = 0.08
    min_energy_ratio: float = 0.35
    band_count: int = DEFAULT_BAND_COUNT
    useful_bins: int = DEFAULT_USEFUL_BINS
    activity_percentile: float = 25.0
    activity_ratio: float = 0.30
    activity_gap_frames: int = 2
    reference_padding_frames: int = 3
    min_reference_frames: int = 8
    max_reference_frames: int = 32
    search_margin_frames: int = 24
    max_search_frames: int = 64


@dataclass(frozen=True)
class ReferenceTemplate:
    band_frames_unit: np.ndarray
    envelope_unit: np.ndarray
    mean_energy: float
    peak_energy: float
    frame_count: int
    start_frame: int
    stop_frame: int


@dataclass(frozen=True)
class ComparisonResult:
    score: float
    spectral_score: float
    envelope_score: float
    energy_score: float
    baseline_score: float
    valid_windows: int
    total_windows: int
    best_index: int
    search_frames: int


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    centered = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + EPSILON
    return centered / norms


def _unit_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    centered = x - np.mean(x, dtype=np.float32)
    norm = np.linalg.norm(centered) + EPSILON
    return centered / norm


def _window_mean_2d(x: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    prefix = np.zeros((x.shape[0] + 1, x.shape[1]), dtype=np.float32)
    prefix[1:] = np.cumsum(x, axis=0, dtype=np.float32)
    window_sum = prefix[window_size:] - prefix[:-window_size]
    return window_sum[::step] / np.float32(window_size)


def _window_mean_1d(x: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    prefix = np.zeros(x.shape[0] + 1, dtype=np.float32)
    prefix[1:] = np.cumsum(x, dtype=np.float32)
    window_sum = prefix[window_size:] - prefix[:-window_size]
    return window_sum[::step] / np.float32(window_size)


def _agrupar_bandas_fft(
    fft_frames: np.ndarray,
    *,
    band_count: int = DEFAULT_BAND_COUNT,
    useful_bins: int = DEFAULT_USEFUL_BINS,
) -> np.ndarray:
    fft_frames = np.asarray(fft_frames, dtype=np.float32)
    if fft_frames.ndim != 2:
        raise ValueError("FFT data must be 2D")

    useful_bins = max(1, min(int(useful_bins), fft_frames.shape[1]))
    useful = np.log1p(np.maximum(fft_frames[:, :useful_bins], 0.0))
    band_count = max(1, min(int(band_count), useful.shape[1]))

    if useful.shape[1] % band_count == 0:
        bins_per_band = useful.shape[1] // band_count
        grouped = useful.reshape(useful.shape[0], band_count, bins_per_band)
        return np.mean(grouped, axis=2, dtype=np.float32)

    edges = np.linspace(0, useful.shape[1], num=band_count + 1, dtype=np.int32)
    bands = np.zeros((useful.shape[0], band_count), dtype=np.float32)
    for idx in range(band_count):
        start = int(edges[idx])
        stop = max(start + 1, int(edges[idx + 1]))
        bands[:, idx] = np.mean(useful[:, start:stop], axis=1, dtype=np.float32)
    return bands


def _energia_frames(bands: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(bands, dtype=np.float32), axis=1, dtype=np.float32)


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or mask.size == 0:
        return mask

    idx = 0
    size = mask.size
    while idx < size:
        if mask[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < size and not mask[idx]:
            idx += 1
        gap_stop = idx
        if gap_start == 0 or gap_stop == size:
            continue
        if (gap_stop - gap_start) <= max_gap:
            mask[gap_start:gap_stop] = True
    return mask


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    mask = np.asarray(mask, dtype=bool)
    best_start = 0
    best_stop = 0
    idx = 0
    while idx < mask.size:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < mask.size and mask[idx]:
            idx += 1
        stop = idx
        if (stop - start) > (best_stop - best_start):
            best_start = start
            best_stop = stop
    return best_start, best_stop


def _highest_energy_window(energy: np.ndarray, window_size: int) -> tuple[int, int]:
    energy = np.asarray(energy, dtype=np.float32)
    if energy.size <= window_size:
        return 0, int(energy.size)

    prefix = np.zeros(energy.size + 1, dtype=np.float32)
    prefix[1:] = np.cumsum(energy, dtype=np.float32)
    window_sum = prefix[window_size:] - prefix[:-window_size]
    start = int(np.argmax(window_sum))
    return start, start + int(window_size)


def _select_reference_slice(energy: np.ndarray, cfg: DirectComparatorConfig) -> tuple[int, int]:
    energy = np.asarray(energy, dtype=np.float32)
    frame_count = energy.size
    if frame_count == 0:
        return 0, 0
    if frame_count <= cfg.min_reference_frames:
        return 0, frame_count

    floor = float(np.percentile(energy, cfg.activity_percentile))
    ceiling = float(np.percentile(energy, 95.0))
    threshold = floor + (cfg.activity_ratio * max(ceiling - floor, EPSILON))

    active = _fill_short_gaps(energy >= threshold, cfg.activity_gap_frames)
    start, stop = _longest_true_run(active)
    if stop <= start:
        peak = int(np.argmax(energy))
        start = max(0, peak - (cfg.min_reference_frames // 2))
        stop = min(frame_count, start + cfg.min_reference_frames)
        start = max(0, stop - cfg.min_reference_frames)

    start = max(0, start - cfg.reference_padding_frames)
    stop = min(frame_count, stop + cfg.reference_padding_frames)

    if (stop - start) < cfg.min_reference_frames:
        peak = int(np.argmax(energy))
        start = max(0, peak - (cfg.min_reference_frames // 2))
        stop = min(frame_count, start + cfg.min_reference_frames)
        start = max(0, stop - cfg.min_reference_frames)

    if (stop - start) > cfg.max_reference_frames:
        rel_start, rel_stop = _highest_energy_window(energy[start:stop], cfg.max_reference_frames)
        start += rel_start
        stop = start + (rel_stop - rel_start)

    return start, stop


def _search_tail_frames(template_len: int, cfg: DirectComparatorConfig) -> int:
    desired = max(template_len + cfg.search_margin_frames, template_len * 2)
    return max(template_len, min(desired, cfg.max_search_frames))


def build_reference_template(
    evento_fft: np.ndarray,
    *,
    config: Optional[DirectComparatorConfig] = None,
) -> ReferenceTemplate:
    cfg = config or DirectComparatorConfig()
    bands = _agrupar_bandas_fft(evento_fft, band_count=cfg.band_count, useful_bins=cfg.useful_bins)
    energy = _energia_frames(bands)
    start, stop = _select_reference_slice(energy, cfg)

    ref_bands = bands[start:stop]
    ref_energy = energy[start:stop]

    return ReferenceTemplate(
        band_frames_unit=_unit_rows(ref_bands),
        envelope_unit=_unit_vector(ref_energy),
        mean_energy=float(np.mean(ref_energy, dtype=np.float32) + EPSILON),
        peak_energy=float(np.max(ref_energy) + EPSILON),
        frame_count=int(ref_bands.shape[0]),
        start_frame=int(start),
        stop_frame=int(stop),
    )


def score_history_against_template(
    snapshot_fft: np.ndarray,
    template: ReferenceTemplate,
    *,
    config: Optional[DirectComparatorConfig] = None,
) -> ComparisonResult:
    cfg = config or DirectComparatorConfig()
    snapshot_fft = np.asarray(snapshot_fft, dtype=np.float32)
    if snapshot_fft.ndim != 2:
        raise ValueError("FFT history must be 2D")
    if snapshot_fft.shape[0] < template.frame_count or template.frame_count <= 0:
        raise ValueError("FFT history shorter than template")

    bands = _agrupar_bandas_fft(snapshot_fft, band_count=cfg.band_count, useful_bins=cfg.useful_bins)
    energy = _energia_frames(bands)

    search_frames = _search_tail_frames(template.frame_count, cfg)
    if bands.shape[0] > search_frames:
        bands = bands[-search_frames:]
        energy = energy[-search_frames:]

    normalized_bands = _unit_rows(bands)
    band_windows = sliding_window_view(normalized_bands, template.frame_count, axis=0)
    band_windows = np.moveaxis(band_windows, -1, 1)
    energy_windows = sliding_window_view(energy, template.frame_count)

    frame_cosine = np.sum(band_windows * template.band_frames_unit[None, :, :], axis=2, dtype=np.float32)
    spectral_score = np.mean(frame_cosine, axis=1, dtype=np.float32)

    energy_centered = energy_windows - np.mean(energy_windows, axis=1, keepdims=True, dtype=np.float32)
    energy_norm = np.linalg.norm(energy_centered, axis=1) + EPSILON
    envelope_score = (energy_centered @ template.envelope_unit) / energy_norm

    mean_energy = np.mean(energy_windows, axis=1, dtype=np.float32)
    peak_energy = np.max(energy_windows, axis=1)
    energy_ratio = np.maximum(mean_energy / np.float32(template.mean_energy), EPSILON)
    energy_score = 1.0 - (np.minimum(np.abs(np.log2(energy_ratio)), 1.5) / 1.5)
    energy_score = np.clip(energy_score, 0.0, 1.0)

    valid = (mean_energy >= (template.mean_energy * cfg.min_energy_ratio)) & (
        peak_energy >= (template.peak_energy * cfg.min_energy_ratio)
    )

    score = (
        0.70 * ((spectral_score + 1.0) * 0.5)
        + 0.20 * ((envelope_score + 1.0) * 0.5)
        + 0.10 * energy_score
    )
    score = np.where(valid, score, 0.0)

    best_index = int(np.argmax(score))
    return ComparisonResult(
        score=float(score[best_index]),
        spectral_score=float(spectral_score[best_index]),
        envelope_score=float(envelope_score[best_index]),
        energy_score=float(energy_score[best_index]),
        baseline_score=float(mean_energy[best_index]),
        valid_windows=int(np.count_nonzero(valid)),
        total_windows=int(score.size),
        best_index=best_index,
        search_frames=int(bands.shape[0]),
    )


def _write_similarity_state(active: bool) -> None:
    with open(SIMILARITY_STATE_TMP_FILENAME, "w", encoding="ascii") as handle:
        handle.write("1\n" if active else "0\n")
    os.replace(SIMILARITY_STATE_TMP_FILENAME, SIMILARITY_STATE_FILENAME)


def compararEvento(buffer2, buffer4, lock, get_lastEventTime, config: Optional[DirectComparatorConfig] = None):
    del buffer2
    cfg = config or DirectComparatorConfig()

    template = None
    detectando = False
    similarity_state = None
    consecutive_hits = 0
    last_event_mtime = 0.0
    last_fft_mtime = 0.0
    recent_scores = deque(maxlen=cfg.score_history_size)

    _write_similarity_state(False)

    while True:
        time.sleep(cfg.poll_interval_seconds)

        if time.time() - get_lastEventTime() < cfg.cooldown_seconds:
            continue

        if not os.path.exists(EVENTO_FILENAME) or not os.path.exists(FFT_FILENAME):
            continue

        event_mtime = os.path.getmtime(EVENTO_FILENAME)
        fft_mtime = os.path.getmtime(FFT_FILENAME)
        if template is None or event_mtime != last_event_mtime or fft_mtime != last_fft_mtime:
            evento_fft = np.load(FFT_FILENAME).astype(np.float32)
            template = build_reference_template(evento_fft, config=cfg)
            last_event_mtime = event_mtime
            last_fft_mtime = fft_mtime
            recent_scores.clear()
            consecutive_hits = 0
            detectando = False
            if similarity_state is not False:
                _write_similarity_state(False)
                similarity_state = False

            print(
                f"referencia atualizada: frames={template.frame_count} "
                f"janela={template.start_frame}:{template.stop_frame}",
                flush=True,
            )

        with lock:
            snapshot_fft = list(buffer4)

        if template is None or len(snapshot_fft) < template.frame_count:
            continue

        loop_start = time.perf_counter()
        result = score_history_against_template(np.asarray(snapshot_fft, dtype=np.float32), template, config=cfg)

        baseline = float(np.median(np.asarray(recent_scores, dtype=np.float32))) if recent_scores else 0.0
        margin = result.score - baseline
        elapsed_ms = (time.perf_counter() - loop_start) * 1000.0

        print(
            f"score={result.score:.3f} "
            f"base={baseline:.3f} "
            f"margem={margin:.3f} "
            f"espectro={result.spectral_score:.3f} "
            f"env={result.envelope_score:.3f} "
            f"energia={result.energy_score:.3f} "
            f"janelas={result.total_windows} "
            f"validas={result.valid_windows} "
            f"ref={template.frame_count} "
            f"busca={result.search_frames} "
            f"proc_ms={elapsed_ms:.1f}",
            flush=True,
        )

        is_hit = result.score >= cfg.absolute_threshold and (not recent_scores or margin >= cfg.margin_threshold)
        if is_hit:
            consecutive_hits += 1
        else:
            consecutive_hits = 0
            recent_scores.append(result.score)

        if consecutive_hits >= cfg.min_consecutive_hits:
            if not detectando:
                print("OPA! SOM SEMELHANTE!!!", flush=True)
                detectando = True
            if similarity_state is not True:
                _write_similarity_state(True)
                similarity_state = True
        else:
            detectando = False
            if similarity_state is not False:
                _write_similarity_state(False)
                similarity_state = False
