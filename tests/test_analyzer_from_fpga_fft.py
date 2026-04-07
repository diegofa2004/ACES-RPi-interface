import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import analyzer_from_fpga_fft
from rpi3b_i2s_fft.i2s_stream import DEFAULT_CAPTURE_RATE_HZ


class AnalyzerFromFPGAFFTTests(unittest.TestCase):
    def test_frames_for_seconds_rounds_up(self):
        self.assertEqual(analyzer_from_fpga_fft.frames_for_seconds(DEFAULT_CAPTURE_RATE_HZ, 512, 5.0), 477)

    def test_arm_recording_only_arms_once(self):
        buffers = analyzer_from_fpga_fft.create_analysis_buffers(8, 2, prebuffer_seconds=1.0, history_seconds=2.0)
        state = analyzer_from_fpga_fft.create_runtime_state()
        self.assertTrue(analyzer_from_fpga_fft.arm_recording(state, 1.0, buffers))
        self.assertFalse(analyzer_from_fpga_fft.arm_recording(state, 2.0, buffers))

    def test_ingest_frame_emits_snapshot_after_record_window(self):
        buffers = analyzer_from_fpga_fft.create_analysis_buffers(
            sample_rate=8,
            frame_bins=2,
            prebuffer_seconds=1.0,
            history_seconds=2.0,
        )
        state = analyzer_from_fpga_fft.create_runtime_state()

        analyzer_from_fpga_fft.ingest_frame(buffers, state, np.arange(8), np.arange(4), 0.0, record_seconds=0.25)
        analyzer_from_fpga_fft.ingest_frame(buffers, state, np.arange(8) + 10, np.arange(4) + 10, 0.2, record_seconds=0.25)

        analyzer_from_fpga_fft.arm_recording(state, 1.0, buffers)
        self.assertIsNone(
            analyzer_from_fpga_fft.ingest_frame(
                buffers,
                state,
                np.arange(8) + 20,
                np.arange(4) + 20,
                1.0,
                record_seconds=0.25,
            )
        )

        event = analyzer_from_fpga_fft.ingest_frame(
            buffers,
            state,
            np.arange(8) + 30,
            np.arange(4) + 30,
            1.3,
            record_seconds=0.25,
        )
        self.assertIsNotNone(event)
        evento, fft = event

        self.assertFalse(state["recording"])
        self.assertEqual(state["last_event_time"], 1.3)
        self.assertEqual(evento.shape, (4, 8))
        self.assertEqual(fft.shape, (4, 4))
        np.testing.assert_array_equal(evento[0], np.arange(8))
        np.testing.assert_array_equal(evento[-1], np.arange(8) + 30)

    def test_ingest_frame_trims_buffers_by_wall_clock(self):
        buffers = analyzer_from_fpga_fft.create_analysis_buffers(
            sample_rate=48828,
            frame_bins=512,
            prebuffer_seconds=1.0,
            history_seconds=2.0,
        )
        state = analyzer_from_fpga_fft.create_runtime_state()

        analyzer_from_fpga_fft.ingest_frame(buffers, state, np.arange(8), np.arange(4), 0.0)
        analyzer_from_fpga_fft.ingest_frame(buffers, state, np.arange(8) + 10, np.arange(4) + 10, 0.8)
        analyzer_from_fpga_fft.ingest_frame(buffers, state, np.arange(8) + 20, np.arange(4) + 20, 1.6)
        analyzer_from_fpga_fft.ingest_frame(buffers, state, np.arange(8) + 30, np.arange(4) + 30, 2.4)

        self.assertEqual(len(buffers["pre_mfcc"]), 2)
        self.assertEqual(len(buffers["pre_fft"]), 2)
        self.assertEqual(len(buffers["history_mfcc"]), 3)
        self.assertEqual(len(buffers["history_fft"]), 3)
        np.testing.assert_array_equal(buffers["pre_mfcc"][0], np.arange(8) + 20)
        np.testing.assert_array_equal(buffers["history_fft"][0], np.arange(4) + 10)


if __name__ == "__main__":
    unittest.main()
