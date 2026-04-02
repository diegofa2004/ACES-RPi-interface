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
        stereo = np.asarray([[0x4000000A, 0x40000014], [0x8003FFFF, 0x80000008]], dtype=np.uint32).view(np.int32)
        timestamps = iter([1000, 1001])
        tracker = fft_i2s_logger.create_contract_tracker(
            frame_bins=2,
            bfpexp_hold_pairs=1,
            allow_fft_without_bfpexp=False,
            loss_tolerance_pairs=0,
        )
        next_seq = fft_i2s_logger.write_csv_rows(
            writer,
            stereo,
            7,
            tag_shift=30,
            tag_mask=0x3,
            payload_bits=18,
            tag_idle=0,
            tag_bfpexp=1,
            tag_fft=2,
            contract_tracker=tracker,
            timestamp_ns_fn=lambda: next(timestamps),
        )

        self.assertEqual(next_seq, 9)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "1000,7,0x4000000A,0x40000014,bfpexp,bfpexp_preamble,0,0,1,1,10,20,0,0,0,0",
                "1001,8,0x8003FFFF,0x80000008,fft,fft_frame,0,0,2,2,-1,8,0,0,0,0",
            ],
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

    def test_classify_tagged_pair_reports_mismatch(self):
        self.assertEqual(
            fft_i2s_logger.classify_tagged_pair(2, 1, tag_idle=0, tag_bfpexp=1, tag_fft=2),
            "tag_mismatch",
        )

    def test_contract_tracker_requires_full_bfpexp_preamble(self):
        tracker = fft_i2s_logger.create_contract_tracker(
            frame_bins=4,
            bfpexp_hold_pairs=3,
            allow_fft_without_bfpexp=False,
            loss_tolerance_pairs=0,
        )

        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "idle"), ("search_idle", 0, -1))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "bfpexp"), ("bfpexp_preamble", 0, 0))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "bfpexp"), ("bfpexp_preamble", 0, 1))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "fft"), ("protocol_wait_bfpexp", 0, -1))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "bfpexp"), ("bfpexp_preamble", 0, 0))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "bfpexp"), ("bfpexp_preamble", 0, 1))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "bfpexp"), ("bfpexp_preamble", 0, 2))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "fft"), ("fft_frame", 0, 0))

    def test_contract_tracker_resets_when_idle_breaks_fft_frame(self):
        tracker = fft_i2s_logger.create_contract_tracker(
            frame_bins=4,
            bfpexp_hold_pairs=1,
            allow_fft_without_bfpexp=False,
            loss_tolerance_pairs=0,
        )

        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "bfpexp"), ("bfpexp_preamble", 0, 0))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "fft"), ("fft_frame", 0, 0))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "fft"), ("fft_frame", 0, 1))
        self.assertEqual(
            fft_i2s_logger.advance_contract_tracker(tracker, "idle"),
            ("protocol_reset_idle", 0, -1),
        )
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "bfpexp"), ("bfpexp_preamble", 0, 0))

    def test_contract_tracker_tolerates_loss_inside_preamble_and_fft_frame(self):
        tracker = fft_i2s_logger.create_contract_tracker(
            frame_bins=4,
            bfpexp_hold_pairs=2,
            allow_fft_without_bfpexp=False,
            loss_tolerance_pairs=1,
        )

        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "bfpexp"), ("bfpexp_preamble", 0, 0))
        self.assertEqual(
            fft_i2s_logger.advance_contract_tracker(tracker, "tag_mismatch"),
            ("bfpexp_preamble_gap", 0, 1),
        )
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "fft"), ("fft_frame", 0, 0))
        self.assertEqual(
            fft_i2s_logger.advance_contract_tracker(tracker, "idle"),
            ("fft_frame_gap", 0, 1),
        )
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "fft"), ("fft_frame", 0, 2))
        self.assertEqual(fft_i2s_logger.advance_contract_tracker(tracker, "fft"), ("fft_frame", 0, 3))


if __name__ == "__main__":
    unittest.main()
