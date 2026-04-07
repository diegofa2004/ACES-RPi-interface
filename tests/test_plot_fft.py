import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_spi_fft import plotFFT


class PlotFFTTests(unittest.TestCase):
    def test_resolve_cli_defaults_prefers_spectrogram_defaults(self):
        args = mock.Mock(
            plot_mode=None,
            mode_preset="spectrogram",
            sample_mode=None,
            output_file=None,
            capture_next_window=None,
            center_zero=None,
        )

        resolved = plotFFT._resolve_cli_defaults(args)

        self.assertEqual(resolved["plot_mode"], "history")
        self.assertEqual(resolved["sample_mode"], "max-energy")
        self.assertEqual(resolved["output_file"], plotFFT.SCRIPT_DIR / "fft_latest.png")
        self.assertFalse(resolved["capture_next_window"])
        self.assertFalse(resolved["center_zero"])

    def test_resolve_cli_defaults_prefers_capture_window_defaults(self):
        args = mock.Mock(
            plot_mode=None,
            mode_preset="capture-window",
            sample_mode=None,
            output_file=None,
            capture_next_window=None,
            center_zero=None,
        )

        resolved = plotFFT._resolve_cli_defaults(args)

        self.assertEqual(resolved["plot_mode"], "window")
        self.assertEqual(resolved["sample_mode"], "latest")
        self.assertEqual(resolved["output_file"], plotFFT.SCRIPT_DIR / "fft_single_window.png")
        self.assertTrue(resolved["capture_next_window"])
        self.assertTrue(resolved["center_zero"])

    def test_load_pyplot_with_agg_is_headless(self):
        _, backend, interactive = plotFFT._load_pyplot("Agg")
        self.assertFalse(interactive)
        self.assertIn(backend, ("Agg", "builtin-png"))

    def test_load_pyplot_falls_back_without_matplotlib(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            plt, backend, interactive = plotFFT._load_pyplot("Agg")

        self.assertIsNone(plt)
        self.assertEqual(backend, "builtin-png")
        self.assertFalse(interactive)

    def test_load_fft_array_returns_none_for_invalid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "broken.npy"
            invalid_path.write_bytes(b"not-a-valid-npy")
            self.assertIsNone(plotFFT._load_fft_array(invalid_path))

    def test_load_fft_array_reshapes_single_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fft_path = Path(tmpdir) / "fft.npy"
            np.save(fft_path, np.arange(8, dtype=np.float32))

            fft_cache = plotFFT._load_fft_array(fft_path)

        self.assertIsNotNone(fft_cache)
        assert fft_cache is not None
        self.assertEqual(fft_cache.shape, (1, 8))

    def test_select_representative_frame_prefers_highest_energy(self):
        fft_cache = np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [8.0, 8.0, 8.0, 8.0],
            ],
            dtype=np.float32,
        )

        frame, idx = plotFFT._select_representative_frame(fft_cache, "max-energy")

        self.assertEqual(idx, 2)
        np.testing.assert_array_equal(frame, fft_cache[2])

    def test_select_window_frame_accepts_explicit_negative_index(self):
        fft_cache = np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )

        frame, idx = plotFFT._select_window_frame(fft_cache, "latest", -1)

        self.assertEqual(idx, 2)
        np.testing.assert_array_equal(frame, fft_cache[2])

    def test_centered_fft_view_builds_negative_and_positive_frequency_axis(self):
        fft_frame = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        freqs_hz, centered_values = plotFFT._centered_fft_view(
            fft_frame,
            rate=8,
            frame_bins=8,
        )

        np.testing.assert_array_equal(freqs_hz, np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(centered_values, np.asarray([40.0, 30.0, 20.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32))

    def test_load_mock_fft_frame_reads_expected_magnitudes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "expected_fft.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "example_idx,bin_idx,expected_real,expected_imag",
                        "0,0,3.0,4.0",
                        "0,1,5.0,12.0",
                        "1,0,8.0,15.0",
                    ]
                ),
                encoding="utf-8",
            )

            frame = plotFFT._load_mock_fft_frame(csv_path, example_idx=0)

        np.testing.assert_allclose(frame, np.asarray([5.0, 13.0], dtype=np.float32))

    def test_render_plot_can_write_png_in_headless_mode(self):
        plt, _, _ = plotFFT._load_pyplot("Agg")
        if plt is None:
            self.skipTest("matplotlib is not available in this environment")
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        spectrum_ax, spectrogram_ax = axes
        fft_cache = np.abs(np.arange(1, 41, dtype=np.float32).reshape(5, 8))

        plotFFT._render_plot(
            spectrum_ax,
            spectrogram_ax,
            fft_cache,
            rate=plotFFT.DEFAULT_CAPTURE_RATE_HZ,
            frame_bins=512,
            max_freq=8000.0,
            step_hz=1000.0,
            sample_mode="max-energy",
        )

        self.assertEqual(len(spectrum_ax.lines), 1)
        self.assertIn("Espectro FFT representativo", spectrum_ax.get_title())
        self.assertEqual(spectrogram_ax.get_xlabel(), "Tempo (frames)")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fft.png"
            fig.savefig(output_path, dpi=100)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_render_single_window_plot_supports_zero_center_and_mock_overlay(self):
        plt, _, _ = plotFFT._load_pyplot("Agg")
        if plt is None:
            self.skipTest("matplotlib is not available in this environment")
        fig, spectrum_ax = plt.subplots(1, 1, figsize=(6, 4))
        fft_frame = np.asarray([1.0, 2.0, 8.0, 2.0], dtype=np.float32)
        mock_frame = np.asarray([1.0, 3.0, 7.0, 3.0], dtype=np.float32)

        plotFFT._render_single_window_plot(
            spectrum_ax,
            fft_frame,
            rate=8,
            frame_bins=8,
            max_freq=4.0,
            amplitude_scale="linear",
            center_zero=True,
            sample_mode="latest",
            frame_index=0,
            mock_fft_frame=mock_frame,
        )

        self.assertEqual(len(spectrum_ax.lines), 2)
        self.assertIn("centrada em 0", spectrum_ax.get_title())
        self.assertEqual(spectrum_ax.get_xlabel(), "Frequencia (Hz, centrada em 0)")
        x_limits = spectrum_ax.get_xlim()
        self.assertLessEqual(x_limits[0], -3.0)
        self.assertGreaterEqual(x_limits[1], 3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fft_single_window.png"
            fig.savefig(output_path, dpi=100)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_render_png_fallback_writes_png(self):
        fft_cache = np.abs(np.arange(1, 41, dtype=np.float32).reshape(5, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fft_builtin.png"
            plotFFT._render_png_fallback(
                fft_cache,
                rate=plotFFT.DEFAULT_CAPTURE_RATE_HZ,
                frame_bins=512,
                max_freq=8000.0,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
