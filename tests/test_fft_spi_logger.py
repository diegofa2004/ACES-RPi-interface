import csv
import io
import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_spi_fft import fft_spi_logger


class FFTSPILoggerTests(unittest.TestCase):
    def test_decode_stereo_frames_returns_pairs(self):
        raw = np.asarray([1, 2, 3, 4], dtype=np.int32).tobytes()
        stereo = fft_spi_logger.decode_stereo_frames(raw)
        np.testing.assert_array_equal(stereo, np.asarray([[1, 2], [3, 4]], dtype=np.int32))

    def test_write_csv_rows_increments_sequence(self):
        output = io.StringIO()
        writer = csv.writer(output)
        stereo = np.asarray([[10, 20], [30, 40]], dtype=np.int32)
        timestamps = iter([1000, 1001])
        next_seq = fft_spi_logger.write_csv_rows(writer, stereo, 7, timestamp_ns_fn=lambda: next(timestamps))

        self.assertEqual(next_seq, 9)
        self.assertEqual(
            output.getvalue().splitlines(),
            ["1000,7,10,20", "1001,8,30,40"],
        )


if __name__ == "__main__":
    unittest.main()
