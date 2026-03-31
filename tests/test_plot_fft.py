import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import plotFFT


class PlotFFTTests(unittest.TestCase):
    def test_load_pyplot_with_agg_is_headless(self):
        _, backend, interactive = plotFFT._load_pyplot("Agg")
        self.assertEqual(backend, "Agg")
        self.assertFalse(interactive)

    def test_load_fft_array_returns_none_for_invalid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "broken.npy"
            invalid_path.write_bytes(b"not-a-valid-npy")
            self.assertIsNone(plotFFT._load_fft_array(invalid_path))

    def test_render_plot_can_write_png_in_headless_mode(self):
        plt, _, _ = plotFFT._load_pyplot("Agg")
        fig, ax = plt.subplots(figsize=(6, 4))
        fft_cache = np.abs(np.arange(1, 41, dtype=np.float32).reshape(5, 8))

        plotFFT._render_plot(ax, fft_cache, rate=48000, frame_bins=512, max_freq=8000.0, step_hz=1000.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fft.png"
            fig.savefig(output_path, dpi=100)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
