import csv
import io
import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import fft_i2s_logger


class FFTI2SLoggerTests(unittest.TestCase):
    def test_decode_stereo_frames_returns_pairs(self):
        raw = np.asarray([1, 2, 3, 4], dtype=np.int32).tobytes()
        stereo = fft_i2s_logger.decode_stereo_frames(raw)
        np.testing.assert_array_equal(stereo, np.asarray([[1, 2], [3, 4]], dtype=np.int32))

    def test_format_i32_hex_preserves_32bit_pattern(self):
        self.assertEqual(fft_i2s_logger.format_i32_hex(10), "0x0000000A")
        self.assertEqual(fft_i2s_logger.format_i32_hex(-1), "0xFFFFFFFF")
        self.assertEqual(fft_i2s_logger.format_i32_hex(np.int32(-2147483648)), "0x80000000")

    def test_write_csv_rows_increments_sequence(self):
        output = io.StringIO()
        writer = csv.writer(output)
        stereo = np.asarray([[10, 20], [-1, 0x12345678]], dtype=np.int32)
        timestamps = iter([1000, 1001])
        next_seq = fft_i2s_logger.write_csv_rows(writer, stereo, 7, timestamp_ns_fn=lambda: next(timestamps))

        self.assertEqual(next_seq, 9)
        self.assertEqual(
            output.getvalue().splitlines(),
            ["1000,7,0x0000000A,0x00000014", "1001,8,0xFFFFFFFF,0x12345678"],
        )

    def test_mirrored_pair_normalizer_keeps_preferred_orientation(self):
        normalizer = fft_i2s_logger.MirroredPairNormalizer()
        stereo = np.asarray(
            [
                [0x80015555, 0x8000AAAB],
                [0x8000AAAB, 0x80015555],
                [0x80015555, 0x8000AAAB],
                [0x40000012, 0x40000012],
            ],
            dtype=np.uint32,
        ).view(np.int32)

        normalized = normalizer.normalize(stereo)
        expected = np.asarray(
            [
                [0x80015555, 0x8000AAAB],
                [0x80015555, 0x8000AAAB],
                [0x80015555, 0x8000AAAB],
                [0x40000012, 0x40000012],
            ],
            dtype=np.uint32,
        ).view(np.int32)
        np.testing.assert_array_equal(normalized, expected)


if __name__ == "__main__":
    unittest.main()
