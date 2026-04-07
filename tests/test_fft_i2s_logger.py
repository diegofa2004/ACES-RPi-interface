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

    def test_select_logged_channel_shifts_selected_mono_data(self):
        stereo = np.asarray(
            [
                [10, 100],
                [20, 200],
                [30, 300],
            ],
            dtype=np.int32,
        )

        mono, used = fft_i2s_logger.select_logged_channel(
            stereo,
            channel_mode="right",
            sample_shift_bits=2,
        )

        self.assertEqual(used, "right")
        np.testing.assert_array_equal(mono, np.asarray([25, 50, 75], dtype=np.int32))

    def test_write_csv_header_and_rows_emit_raw_and_mono_columns(self):
        output = io.StringIO()
        writer = csv.writer(output)
        fft_i2s_logger.write_csv_header(writer)
        stereo = np.asarray([[10, -20], [30, -40]], dtype=np.int32)
        mono = np.asarray([10, 30], dtype=np.int32)
        timestamps = iter([1000, 1001])

        next_seq = fft_i2s_logger.write_csv_rows(
            writer,
            stereo,
            mono,
            "left",
            7,
            timestamp_ns_fn=lambda: next(timestamps),
        )

        self.assertEqual(next_seq, 9)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "timestamp_ns,sequence,left_i32,right_i32,left_hex,right_hex,mono_i32,channel_used,abs_left,abs_right,abs_mono",
                "1000,7,10,-20,0x0000000A,0xFFFFFFEC,10,left,10,20,10",
                "1001,8,30,-40,0x0000001E,0xFFFFFFD8,30,left,30,40,30",
            ],
        )

    def test_write_csv_rows_requires_matching_frame_count(self):
        writer = csv.writer(io.StringIO())
        with self.assertRaises(ValueError):
            fft_i2s_logger.write_csv_rows(
                writer,
                np.asarray([[1, 2], [3, 4]], dtype=np.int32),
                np.asarray([1], dtype=np.int32),
                "left",
                0,
            )


if __name__ == "__main__":
    unittest.main()
