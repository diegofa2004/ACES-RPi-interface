import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import analyzer_from_fpga_fft
from rpi3b_i2s_fft.fpga_fft_adapter import FFTAdapterConfig
from tests.test_support import pack_tagged_pairs, pack_tagged_word


class AnalyzerFromFPGAFFTTests(unittest.TestCase):
    def test_process_channel_debug_chunk_reports_kinds_and_fft_runs(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, use_i2s_tags=True)
        pairs = np.frombuffer(
            pack_tagged_pairs(
                [
                    (0, 0, 0),
                    (1, 7, 7),
                    (2, 3, 4),
                    (2, 5, 12),
                    (0, 0, 0),
                ]
            ),
            dtype=np.int32,
        ).reshape(-1, 2)
        state = analyzer_from_fpga_fft.create_channel_debug_state()

        chunk = analyzer_from_fpga_fft.process_channel_debug_chunk(
            pairs,
            cfg,
            state,
            preview_pairs=3,
            flag_active=True,
            timestamp_ns=123,
        )
        analyzer_from_fpga_fft.finalize_channel_debug_state(state)
        summary = analyzer_from_fpga_fft.build_channel_debug_summary(
            state,
            duration_seconds=1.5,
            interrupted=False,
            timestamp_ns=456,
        )

        self.assertEqual(chunk["pair_count"], 5)
        self.assertEqual(chunk["kind_counts"]["idle"], 2)
        self.assertEqual(chunk["kind_counts"]["bfpexp"], 1)
        self.assertEqual(chunk["kind_counts"]["fft"], 2)
        self.assertEqual(chunk["fft_run_lengths"], [2])
        self.assertEqual(len(chunk["preview"]), 3)
        self.assertEqual(summary["top_fft_run_lengths"], [2])
        self.assertEqual(summary["flag_high_chunks"], 1)

    def test_process_channel_debug_chunk_counts_mismatch_and_reserved_bits(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, use_i2s_tags=True)
        left_word = pack_tagged_word(2, 5)
        right_word = pack_tagged_word(1, 5)
        reserved_word = int(np.asarray([np.uint32((2 << 30) | (1 << 18) | 9)], dtype=np.uint32).view(np.int32)[0])
        pairs = np.asarray(
            [
                [left_word, right_word],
                [reserved_word, reserved_word],
            ],
            dtype=np.int32,
        )
        state = analyzer_from_fpga_fft.create_channel_debug_state()

        chunk = analyzer_from_fpga_fft.process_channel_debug_chunk(
            pairs,
            cfg,
            state,
            preview_pairs=2,
            flag_active=None,
            timestamp_ns=789,
        )
        analyzer_from_fpga_fft.finalize_channel_debug_state(state)
        summary = analyzer_from_fpga_fft.build_channel_debug_summary(
            state,
            duration_seconds=0.5,
            interrupted=True,
            timestamp_ns=1011,
        )

        self.assertEqual(chunk["kind_counts"]["tag_mismatch"], 1)
        self.assertEqual(chunk["kind_counts"]["fft"], 1)
        self.assertEqual(chunk["reserved_nonzero_words"], 2)
        self.assertEqual(summary["reserved_nonzero_words"], 2)
        self.assertEqual(summary["flag_unknown_chunks"], 1)

    def test_frames_for_seconds_rounds_up(self):
        self.assertEqual(analyzer_from_fpga_fft.frames_for_seconds(48000, 512, 5.0), 469)

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


if __name__ == "__main__":
    unittest.main()
