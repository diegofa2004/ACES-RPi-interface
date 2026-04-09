import sys
import unittest
from pathlib import Path

import numpy as np


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import live_spectrogram


class LiveSpectrogramTests(unittest.TestCase):
    def test_update_live_figure_avoids_degenerate_frequency_axis_for_single_bin(self):
        plt, _, _ = live_spectrogram._load_pyplot("Agg")
        fig, spectrum_ax, spectrogram_ax, spectrum_line, spectrogram_im = live_spectrogram._create_live_figure(plt)

        peak_freq_hz, history_duration = live_spectrogram._update_live_figure(
            spectrum_ax,
            spectrogram_ax,
            spectrum_line,
            spectrogram_im,
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            rate=8,
            frame_bins=8,
            max_freq=1.0,
            step_hz=1.0,
            history_seconds=2.0,
            frame_duration_seconds=1.0,
            dynamic_range_db=40.0,
            min_db=None,
            max_db=None,
            smoothed_frames=1,
            freq_scale="linear",
        )

        x_limits = spectrum_ax.get_xlim()
        y_limits = spectrogram_ax.get_ylim()

        self.assertGreater(x_limits[1], x_limits[0])
        self.assertGreater(y_limits[1], y_limits[0])
        self.assertEqual(peak_freq_hz, 0.0)
        self.assertGreaterEqual(history_duration, 0.0)
        fig.clf()


if __name__ == "__main__":
    unittest.main()
