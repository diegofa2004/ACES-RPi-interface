import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import sine_1k_reference


class Sine1KReferenceTests(unittest.TestCase):
    def test_reference_arrays_have_expected_shape(self):
        self.assertEqual(sine_1k_reference.FFT_REAL.shape, (sine_1k_reference.FRAME_BINS,))
        self.assertEqual(sine_1k_reference.FFT_IMAG.shape, (sine_1k_reference.FRAME_BINS,))
        self.assertEqual(sine_1k_reference.TRANSPORT_LEFT_WORDS.shape, (sine_1k_reference.FRAME_BINS,))
        self.assertEqual(sine_1k_reference.TRANSPORT_RIGHT_WORDS.shape, (sine_1k_reference.FRAME_BINS,))

    def test_peak_bin_stays_near_1k(self):
        magnitude = sine_1k_reference.build_magnitude_frame()
        peak_bin = int(np.argmax(magnitude))
        self.assertEqual(peak_bin, sine_1k_reference.EXPECTED_PEAK_BIN)
        self.assertAlmostEqual(sine_1k_reference.EXPECTED_PEAK_HZ, 953.671875, places=6)

    def test_write_fft_npy_creates_repeated_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fft.npy"
            written = sine_1k_reference.write_fft_npy(output_path, frame_count=5)
            self.assertEqual(written, output_path)
            fft_history = np.load(output_path)

        self.assertEqual(fft_history.shape, (5, sine_1k_reference.USEFUL_BINS))
        np.testing.assert_allclose(fft_history[0], fft_history[-1])


if __name__ == "__main__":
    unittest.main()
