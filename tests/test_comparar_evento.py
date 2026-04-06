import sys
import unittest
import importlib
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

comparar_evento_module = importlib.import_module("rpi3b_spi_fft.compararEvento")


class CompararEventoHelperTests(unittest.TestCase):
    @staticmethod
    def _make_fft_event(peak_band: int, *, background: float, peak_gain: float) -> np.ndarray:
        frames = 16
        bins_per_band = 8
        fft = np.full((frames, 256), background, dtype=np.float32)
        profile = np.asarray([0, 0, 1, 4, 8, 12, 8, 4, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        start = peak_band * bins_per_band
        stop = start + bins_per_band
        fft[:, start:stop] += (profile[:, None] * np.float32(peak_gain))
        return fft

    def test_window_mean_helpers_match_naive_reference(self):
        data_1d = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
        data_2d = np.asarray([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)

        mean_1d = comparar_evento_module._window_mean_1d(data_1d, 3, 1)
        mean_2d = comparar_evento_module._window_mean_2d(data_2d, 2, 1)

        np.testing.assert_allclose(mean_1d, np.asarray([2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(
            mean_2d,
            np.asarray([[1.5, 15.0], [2.5, 25.0], [3.5, 35.0]], dtype=np.float32),
        )

    def test_melhor_bloco_continuo_tracks_high_energy_region(self):
        energia = np.zeros(40, dtype=np.float32)
        energia[12:20] = np.asarray([2, 4, 6, 8, 8, 6, 4, 2], dtype=np.float32)
        i0, i1 = comparar_evento_module._melhor_bloco_continuo(energia, min_frac=0.2, max_frac=0.4)
        self.assertGreaterEqual(i0, 12)
        self.assertLessEqual(i1, 20)

    def test_agrupar_bandas_fft_reduces_256_bins_to_32_band_means(self):
        fft = np.tile(np.arange(256, dtype=np.float32), (2, 1))
        band_means = comparar_evento_module._agrupar_bandas_fft(fft)

        esperado = np.asarray(
            [np.mean(np.arange(idx * 8, (idx + 1) * 8), dtype=np.float32) for idx in range(32)],
            dtype=np.float32,
        )
        self.assertEqual(band_means.shape, (2, 32))
        np.testing.assert_allclose(band_means[0], esperado)

    def test_suprime_ruido_estacionario_highlights_distinctive_band(self):
        bands = np.asarray(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 6.0, 2.0],
                [2.0, 2.0, 8.0, 2.0],
                [2.0, 2.0, 6.0, 2.0],
            ],
            dtype=np.float32,
        )

        filtered = comparar_evento_module._suprime_ruido_estacionario(bands, percentile=40.0)
        np.testing.assert_allclose(filtered[:, 0], 0.0)
        np.testing.assert_allclose(filtered[:, 1], 0.0)
        np.testing.assert_allclose(filtered[:, 3], 0.0)
        self.assertGreater(float(np.max(filtered[:, 2])), 0.0)

    def test_noise_robust_signatures_prefer_matching_event_over_stationary_noise(self):
        ref_fft = self._make_fft_event(peak_band=6, background=1.0, peak_gain=3.0)
        candidate_fft = self._make_fft_event(peak_band=6, background=7.0, peak_gain=3.0)
        distractor_fft = self._make_fft_event(peak_band=18, background=7.0, peak_gain=3.0)

        ref_prepared = comparar_evento_module._prepara_fft(ref_fft)
        ref_fft_signature, ref_weights = comparar_evento_module._assinatura_fft(ref_prepared)
        ref_flux_signature = comparar_evento_module._assinatura_fluxo(ref_prepared)

        def best_similarity(fft_frames: np.ndarray) -> float:
            prepared = comparar_evento_module._prepara_fft(fft_frames)
            bands = comparar_evento_module._prepara_fft_bandas(prepared)
            fft_windows = comparar_evento_module._window_mean_2d(bands, ref_fft.shape[0], 1)
            flux = comparar_evento_module._fluxo_espectral_bandas(bands)
            flux_windows = comparar_evento_module._window_mean_2d(flux, ref_fft.shape[0] - 1, 1)

            s_fft = comparar_evento_module._cosine_batch_weighted(
                comparar_evento_module._zscore_rows(fft_windows),
                ref_fft_signature,
                ref_weights,
            )
            s_flux = comparar_evento_module._cosine_batch(
                comparar_evento_module._zscore_rows(flux_windows),
                ref_flux_signature,
            )
            return float(np.max((0.6 * s_fft) + (0.4 * s_flux)))

        matching_score = best_similarity(candidate_fft)
        distractor_score = best_similarity(distractor_fft)

        self.assertGreater(matching_score, distractor_score + 0.20)


if __name__ == "__main__":
    unittest.main()
