import sys
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft.verify_transport_stream import TaggedTransportConfig, analyze_tagged_hex_stream
from tests.test_support import pack_tagged_word


def _pair_line(tag: int, left: int, right: int, *, payload_bits: int = 18, tag_shift: int = 30) -> str:
    left_word = pack_tagged_word(tag, left, payload_bits=payload_bits, tag_shift=tag_shift)
    right_word = pack_tagged_word(tag, right, payload_bits=payload_bits, tag_shift=tag_shift)
    return f"0x{left_word & 0xFFFFFFFF:08X} 0x{right_word & 0xFFFFFFFF:08X}\n"


class VerifyTransportStreamTests(unittest.TestCase):
    def test_detects_full_frames_and_zero_shift_when_frames_match(self):
        cfg = TaggedTransportConfig(frame_bins=4, useful_bins=4, bfpexp_pairs_required=2)
        lines = [
            _pair_line(0, 0, 0),
            _pair_line(0, 0, 0),
            _pair_line(1, 4, 4),
            _pair_line(1, 4, 4),
            _pair_line(2, 1, 2),
            _pair_line(2, 3, 4),
            _pair_line(2, 5, 12),
            _pair_line(2, 8, 15),
            _pair_line(0, 0, 0),
            _pair_line(1, 4, 4),
            _pair_line(1, 4, 4),
            _pair_line(2, 1, 2),
            _pair_line(2, 3, 4),
            _pair_line(2, 5, 12),
            _pair_line(2, 8, 15),
        ]

        summary = analyze_tagged_hex_stream(lines, cfg, print_limit=0, printer=lambda _line: None)

        self.assertEqual(len(summary.frame_reports), 2)
        self.assertEqual(summary.frame_reports[0].peak_bin, 3)
        self.assertEqual(summary.frame_reports[1].shift_from_prev, 0)
        self.assertIn((2, 4), summary.bfpexp_to_fft_examples)

    def test_detects_circular_shift_between_two_full_frames(self):
        cfg = TaggedTransportConfig(frame_bins=4, useful_bins=4, bfpexp_pairs_required=2)
        lines = [
            _pair_line(1, 4, 4),
            _pair_line(1, 4, 4),
            _pair_line(2, 1, 2),
            _pair_line(2, 3, 4),
            _pair_line(2, 5, 12),
            _pair_line(2, 8, 15),
            _pair_line(0, 0, 0),
            _pair_line(1, 4, 4),
            _pair_line(1, 4, 4),
            _pair_line(2, 3, 4),
            _pair_line(2, 5, 12),
            _pair_line(2, 8, 15),
            _pair_line(2, 1, 2),
        ]

        summary = analyze_tagged_hex_stream(lines, cfg, print_limit=0, printer=lambda _line: None)

        self.assertEqual(len(summary.frame_reports), 2)
        self.assertEqual(summary.frame_reports[1].shift_from_prev, -1)


if __name__ == "__main__":
    unittest.main()
