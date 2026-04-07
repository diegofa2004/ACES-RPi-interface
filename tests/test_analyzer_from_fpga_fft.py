import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_spi_fft import analyzer_from_fpga_fft


class AnalyzerFromFPGAFFTTests(unittest.TestCase):
    def test_frames_for_seconds_rounds_up(self):
        self.assertEqual(analyzer_from_fpga_fft.frames_for_seconds(48_000, 512, 0.001), 1)
        self.assertEqual(analyzer_from_fpga_fft.frames_for_seconds(48_000, 512, 1.0), 94)

    def test_ingest_frame_trims_time_windows(self):
        buffers = analyzer_from_fpga_fft.create_analysis_buffers(
            4,
            1,
            prebuffer_seconds=2.0,
            history_seconds=4.0,
        )
        state = analyzer_from_fpga_fft.create_runtime_state()

        analyzer_from_fpga_fft.ingest_frame(
            buffers,
            state,
            np.arange(13, dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
            1.0,
        )
        analyzer_from_fpga_fft.ingest_frame(
            buffers,
            state,
            np.arange(13, dtype=np.float32) + 10.0,
            np.asarray([3.0, 4.0], dtype=np.float32),
            2.0,
        )
        analyzer_from_fpga_fft.ingest_frame(
            buffers,
            state,
            np.arange(13, dtype=np.float32) + 20.0,
            np.asarray([5.0, 6.0], dtype=np.float32),
            4.5,
        )

        self.assertEqual(len(buffers["pre_mfcc"]), 1)
        self.assertEqual(len(buffers["history_mfcc"]), 3)
        np.testing.assert_array_equal(buffers["pre_mfcc"][0], np.arange(8, dtype=np.float32) + 20.0)

    def test_recording_combines_prebuffer_and_future_frames(self):
        buffers = analyzer_from_fpga_fft.create_analysis_buffers(
            4,
            1,
            prebuffer_seconds=5.0,
            history_seconds=10.0,
        )
        state = analyzer_from_fpga_fft.create_runtime_state()

        analyzer_from_fpga_fft.ingest_frame(
            buffers,
            state,
            np.arange(13, dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
            10.0,
        )
        analyzer_from_fpga_fft.ingest_frame(
            buffers,
            state,
            np.arange(13, dtype=np.float32) + 100.0,
            np.asarray([3.0, 4.0], dtype=np.float32),
            11.0,
        )

        armed = analyzer_from_fpga_fft.arm_recording(state, 11.0, buffers)

        self.assertTrue(armed)

        complete = analyzer_from_fpga_fft.ingest_frame(
            buffers,
            state,
            np.arange(13, dtype=np.float32) + 200.0,
            np.asarray([5.0, 6.0], dtype=np.float32),
            12.0,
            record_seconds=1.0,
        )

        self.assertIsNotNone(complete)
        evento, fft = complete
        self.assertEqual(evento.shape, (3, 8))
        self.assertEqual(fft.shape, (3, 2))
        np.testing.assert_array_equal(evento[0], np.arange(8, dtype=np.float32))
        np.testing.assert_array_equal(evento[1], np.arange(8, dtype=np.float32) + 100.0)
        np.testing.assert_array_equal(evento[2], np.arange(8, dtype=np.float32) + 200.0)
        self.assertFalse(state["recording"])


if __name__ == "__main__":
    unittest.main()
