from typing import Optional, Union

import numpy as np


def hz_to_mel(hz: Union[np.ndarray, float]) -> np.ndarray:
    hz_array = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + (hz_array / 700.0))


def mel_to_hz(mel: Union[np.ndarray, float]) -> np.ndarray:
    mel_array = np.asarray(mel, dtype=np.float64)
    return 700.0 * ((10.0 ** (mel_array / 2595.0)) - 1.0)


def build_mel_filter(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 32,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if n_fft < 2:
        raise ValueError("n_fft must be at least 2")
    if n_mels <= 0:
        raise ValueError("n_mels must be positive")

    nyquist = sample_rate / 2.0
    high_freq = nyquist if fmax is None else float(fmax)
    if not 0.0 <= fmin < high_freq <= nyquist:
        raise ValueError("Expected 0 <= fmin < fmax <= sample_rate / 2")

    fft_freqs = np.linspace(0.0, nyquist, 1 + (n_fft // 2), dtype=np.float64)
    mel_edges = np.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(high_freq)), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)

    filters = np.zeros((n_mels, fft_freqs.size), dtype=np.float32)
    for idx in range(n_mels):
        left = hz_edges[idx]
        center = hz_edges[idx + 1]
        right = hz_edges[idx + 2]

        if center <= left:
            center = np.nextafter(left, np.inf)
        if right <= center:
            right = np.nextafter(center, np.inf)

        lower_slope = (fft_freqs - left) / (center - left)
        upper_slope = (right - fft_freqs) / (right - center)
        filters[idx, :] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))

        width = right - left
        if width > 0.0:
            filters[idx, :] *= 2.0 / width

    return filters


def build_dct_matrix(input_size: int, output_size: int) -> np.ndarray:
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if not 0 < output_size <= input_size:
        raise ValueError("Expected 0 < output_size <= input_size")

    sample_positions = np.arange(input_size, dtype=np.float64) + 0.5
    coeff_positions = np.arange(output_size, dtype=np.float64)[:, None]

    basis = np.cos((np.pi / input_size) * coeff_positions * sample_positions[None, :])
    basis[0, :] *= np.sqrt(1.0 / input_size)
    if output_size > 1:
        basis[1:, :] *= np.sqrt(2.0 / input_size)
    return basis.astype(np.float32)
