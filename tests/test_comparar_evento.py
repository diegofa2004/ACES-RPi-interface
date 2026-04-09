import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

comparar_evento_module = importlib.import_module("rpi3b_i2s_fft.compararEvento")


class CompararEventoHelperTests(unittest.TestCase):
    @staticmethod
    def _make_fft_pattern(
        peak_band: int,
        *,
        total_frames: int = 40,
        active_start: int = 10,
        active_stop: int = 26,
        background: float = 0.5,
        peak_gain: float = 16.0,
        broadband_noise: float = 0.0,
        striped_noise: float = 0.0,
    ) -> np.ndarray:
        fft = np.full((total_frames, 256), background, dtype=np.float32)
        bins_per_band = 8
        envelope = np.asarray([0, 0, 1, 3, 6, 9, 12, 14, 12, 9, 6, 3, 1, 0, 0, 0], dtype=np.float32)
        start = peak_band * bins_per_band
        stop = start + bins_per_band
        fft[active_start:active_stop, start:stop] += envelope[:, None] * np.float32(peak_gain)
        if broadband_noise > 0.0:
            rng = np.random.default_rng(0)
            fft += rng.uniform(0.0, broadband_noise, size=fft.shape).astype(np.float32)
        if striped_noise > 0.0:
            rng = np.random.default_rng(1)
            fft += rng.uniform(0.0, striped_noise, size=(total_frames, 1)).astype(np.float32)
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

    def test_agrupar_bandas_fft_reduces_256_bins_to_32_band_means(self):
        fft = np.tile(np.arange(256, dtype=np.float32), (2, 1))
        band_means = comparar_evento_module._agrupar_bandas_fft(fft)

        esperado = np.asarray(
            [np.mean(np.log1p(np.arange(idx * 8, (idx + 1) * 8)), dtype=np.float32) for idx in range(32)],
            dtype=np.float32,
        )
        self.assertEqual(band_means.shape, (2, 32))
        np.testing.assert_allclose(band_means[0], esperado)

    def test_apply_min_db_gate_zeroes_bins_below_requested_floor(self):
        fft = np.asarray([[1e-4, 1e-2, 1e0]], dtype=np.float32)

        gated = comparar_evento_module._apply_min_db_gate(fft, -20.0, dynamic_range_db=50.0)

        np.testing.assert_allclose(gated, np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))

    def test_apply_min_db_gate_uses_dynamic_range_when_floor_is_not_fixed(self):
        fft = np.asarray([[1e-4, 1e-2, 1e0]], dtype=np.float32)

        gated = comparar_evento_module._apply_min_db_gate(fft, None, dynamic_range_db=50.0)

        np.testing.assert_allclose(gated, np.asarray([[0.0, 1e-2, 1.0]], dtype=np.float32))

    def test_build_reference_template_extracts_active_slice(self):
        cfg = comparar_evento_module.DirectComparatorConfig(
            min_reference_frames=8,
            max_reference_frames=24,
            reference_padding_frames=2,
        )
        fft = self._make_fft_pattern(peak_band=6)

        template = comparar_evento_module.build_reference_template(fft, config=cfg)

        self.assertGreaterEqual(template.start_frame, 8)
        self.assertLessEqual(template.start_frame, 12)
        self.assertGreaterEqual(template.stop_frame, 24)
        self.assertLessEqual(template.stop_frame, 28)
        self.assertGreaterEqual(template.frame_count, 8)
        self.assertLessEqual(template.frame_count, 24)
        self.assertEqual(template.band_frames_salience.shape[0], template.frame_count)
        self.assertGreater(float(np.max(template.band_frames_salience)), 0.0)

    def test_score_history_against_template_prefers_matching_signal(self):
        cfg = comparar_evento_module.DirectComparatorConfig(
            absolute_threshold=0.70,
            max_search_frames=64,
            max_reference_frames=24,
        )
        reference_fft = self._make_fft_pattern(peak_band=6)
        template = comparar_evento_module.build_reference_template(reference_fft, config=cfg)

        history_match = np.concatenate(
            [
                np.full((24, 256), 0.4, dtype=np.float32),
                self._make_fft_pattern(peak_band=6, total_frames=32, active_start=8, active_stop=24, peak_gain=18.0),
            ],
            axis=0,
        )
        history_distractor = np.concatenate(
            [
                np.full((24, 256), 0.4, dtype=np.float32),
                self._make_fft_pattern(peak_band=18, total_frames=32, active_start=8, active_stop=24, peak_gain=18.0),
            ],
            axis=0,
        )

        matching = comparar_evento_module.score_history_against_template(history_match, template, config=cfg)
        distractor = comparar_evento_module.score_history_against_template(history_distractor, template, config=cfg)

        self.assertGreater(matching.score, 0.70)
        self.assertGreater(matching.score, distractor.score + 0.20)
        self.assertGreater(matching.spectral_score, distractor.spectral_score)
        self.assertGreater(matching.valid_windows, 0)

    def test_score_history_against_template_rejects_low_energy_tail(self):
        cfg = comparar_evento_module.DirectComparatorConfig(min_energy_ratio=0.50)
        reference_fft = self._make_fft_pattern(peak_band=4, peak_gain=20.0)
        template = comparar_evento_module.build_reference_template(reference_fft, config=cfg)
        silence = np.full((64, 256), 0.05, dtype=np.float32)

        result = comparar_evento_module.score_history_against_template(silence, template, config=cfg)

        self.assertEqual(result.valid_windows, 0)
        self.assertLess(result.score, 0.05)

    def test_score_history_against_template_rejects_broadband_noise_with_matching_envelope(self):
        cfg = comparar_evento_module.DirectComparatorConfig(
            absolute_threshold=0.70,
            max_search_frames=64,
            max_reference_frames=24,
        )
        reference_fft = self._make_fft_pattern(
            peak_band=6,
            broadband_noise=4.0,
            striped_noise=2.0,
        )
        template = comparar_evento_module.build_reference_template(reference_fft, config=cfg)

        matching_history = np.concatenate(
            [
                np.full((24, 256), 0.4, dtype=np.float32),
                self._make_fft_pattern(
                    peak_band=6,
                    total_frames=32,
                    active_start=8,
                    active_stop=24,
                    peak_gain=18.0,
                    broadband_noise=4.0,
                    striped_noise=2.0,
                ),
            ],
            axis=0,
        )
        broadband_distractor = np.concatenate(
            [
                np.full((24, 256), 0.4, dtype=np.float32),
                self._make_fft_pattern(
                    peak_band=6,
                    total_frames=32,
                    active_start=8,
                    active_stop=24,
                    peak_gain=0.0,
                    broadband_noise=8.0,
                    striped_noise=6.0,
                ),
            ],
            axis=0,
        )

        matching = comparar_evento_module.score_history_against_template(matching_history, template, config=cfg)
        distractor = comparar_evento_module.score_history_against_template(broadband_distractor, template, config=cfg)

        self.assertGreater(matching.score, cfg.absolute_threshold)
        self.assertLess(distractor.score, cfg.absolute_threshold)
        self.assertGreater(matching.score, distractor.score + 0.20)
        self.assertGreater(matching.dominant_score, distractor.dominant_score + 0.20)

    def test_score_history_against_template_handles_heavy_noise_with_band_floor_rejection(self):
        cfg = comparar_evento_module.DirectComparatorConfig(
            absolute_threshold=0.45,
            max_search_frames=64,
            max_reference_frames=24,
            noise_floor_percentile=25.0,
            salience_floor_percentile=40.0,
        )
        reference_fft = self._make_fft_pattern(
            peak_band=6,
            broadband_noise=6.0,
            striped_noise=4.0,
        )
        template = comparar_evento_module.build_reference_template(reference_fft, config=cfg)

        matching_history = np.concatenate(
            [
                np.full((24, 256), 0.4, dtype=np.float32),
                self._make_fft_pattern(
                    peak_band=6,
                    total_frames=32,
                    active_start=8,
                    active_stop=24,
                    peak_gain=18.0,
                    broadband_noise=6.0,
                    striped_noise=4.0,
                ),
            ],
            axis=0,
        )
        broadband_distractor = np.concatenate(
            [
                np.full((24, 256), 0.4, dtype=np.float32),
                self._make_fft_pattern(
                    peak_band=6,
                    total_frames=32,
                    active_start=8,
                    active_stop=24,
                    peak_gain=0.0,
                    broadband_noise=10.0,
                    striped_noise=8.0,
                ),
            ],
            axis=0,
        )

        matching = comparar_evento_module.score_history_against_template(matching_history, template, config=cfg)
        distractor = comparar_evento_module.score_history_against_template(broadband_distractor, template, config=cfg)

        self.assertGreater(matching.score, cfg.absolute_threshold)
        self.assertLess(distractor.score, cfg.absolute_threshold)
        self.assertGreater(matching.score, distractor.score + 0.25)
        self.assertGreater(matching.dominant_score, distractor.dominant_score + 0.15)


if __name__ == "__main__":
    unittest.main()
