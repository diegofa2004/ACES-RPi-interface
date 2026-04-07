import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft.verify_transport_stream import (
    RawTransportConfig,
    analyze_raw_stereo_frames,
    load_raw_capture,
    summary_to_dict,
)


class VerifyTransportStreamTests(unittest.TestCase):
    def test_detects_dominant_channel_and_peak_frequency(self):
        frame_bins = 8
        useful_bins = 5
        left = np.asarray([0, 1, 0, -1, 0, 1, 0, -1], dtype=np.int32)
        right = left * 100
        stereo = np.column_stack((left, right))
        cfg = RawTransportConfig(
            sample_rate=8000,
            frame_bins=frame_bins,
            useful_bins=useful_bins,
            channel_mode="auto",
            remove_dc=False,
        )

        summary = analyze_raw_stereo_frames(stereo, cfg, print_limit=0, printer=lambda _line: None)

        self.assertEqual(summary.sample_count, 8)
        self.assertEqual(summary.window_count, 1)
        self.assertEqual(summary.dominant_channel, "right")
        self.assertEqual(summary.channel_counts["right"], 1)
        self.assertAlmostEqual(summary.peak_frequency_hz, 2000.0, places=3)
        self.assertGreater(summary.right_mean_abs, summary.left_mean_abs)
        self.assertAlmostEqual(summary.mean_correlation_lr, 1.0, places=6)

    def test_load_raw_capture_and_summary_dict_round_trip(self):
        stereo = np.asarray([[100, -100], [50, -50], [25, -25], [0, 0]], dtype=np.int32)
        cfg = RawTransportConfig(
            sample_rate=4000,
            frame_bins=4,
            useful_bins=3,
            channel_mode="left",
            remove_dc=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "capture.raw"
            raw_path.write_bytes(stereo.tobytes())
            loaded = load_raw_capture(raw_path)

        np.testing.assert_array_equal(loaded, stereo)
        summary = analyze_raw_stereo_frames(loaded, cfg, print_limit=0, printer=lambda _line: None)
        payload = summary_to_dict(summary)

        self.assertEqual(payload["sample_count"], 4)
        self.assertEqual(payload["window_count"], 1)
        self.assertEqual(len(payload["frame_reports"]), 1)
        self.assertEqual(payload["frame_reports"][0]["channel_used"], "left")


if __name__ == "__main__":
    unittest.main()
