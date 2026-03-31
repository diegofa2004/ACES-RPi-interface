import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import fpga_fft_adapter
from rpi3b_i2s_fft.fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
from tests.test_support import FakeProcess, pack_raw_pairs, pack_tagged_pairs


class FFTAdapterConfigTests(unittest.TestCase):
    def test_rejects_overlapping_tag_and_payload(self):
        with self.assertRaises(ValueError):
            FFTAdapterConfig(use_i2s_tags=True, tag_shift=10, payload_bits=18)


class FPGAFFTReceiverTests(unittest.TestCase):
    def test_receiver_polls_at_least_one_frame_per_read(self):
        cfg = FFTAdapterConfig(frame_bins=512, useful_bins=256)
        rx = FPGAFFTReceiver(cfg)

        self.assertEqual(rx._poll_pairs, 512)
        self.assertEqual(rx._poll_bytes, 4096)

    def test_start_and_stop_use_capture_process_helpers(self):
        fake_proc = FakeProcess(b"")
        cfg = FFTAdapterConfig(device="auto")
        rx = FPGAFFTReceiver(cfg)

        with mock.patch.object(fpga_fft_adapter, "resolve_audio_device", return_value="hw:2,0"), \
             mock.patch.object(fpga_fft_adapter, "start_arecord_process", return_value=fake_proc) as start_mock, \
             mock.patch.object(fpga_fft_adapter, "stop_process") as stop_mock:
            rx.start()
            start_mock.assert_called_once_with("hw:2,0", 48000)
            rx.stop()
            stop_mock.assert_called_once_with(fake_proc)

    def test_read_frame_in_tagged_mode_decodes_bfpexp_then_fft_pairs(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, use_i2s_tags=True, handshake_timeout_seconds=0.01)
        rx = FPGAFFTReceiver(cfg)
        stream = pack_tagged_pairs(
            [
                (0, 0, 0),
                (1, 7, 7),
                (1, 7, 7),
                (2, 3, 4),
                (2, 5, 12),
                (2, -8, 15),
                (2, 7, -24),
            ]
        )
        rx._proc = FakeProcess(stream)

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, mfcc = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))
        self.assertEqual(mfcc.shape, (13,))

    def test_tagged_mode_requires_bfpexp_by_default_and_can_be_relaxed(self):
        stream = pack_tagged_pairs(
            [
                (2, 1, 2),
                (2, 3, 4),
                (2, 5, 12),
                (2, 8, 15),
            ]
        )

        strict_rx = FPGAFFTReceiver(
            FFTAdapterConfig(frame_bins=4, useful_bins=4, use_i2s_tags=True, handshake_timeout_seconds=0.01)
        )
        strict_rx._proc = FakeProcess(stream)
        self.assertIsNone(strict_rx.read_frame())

        relaxed_rx = FPGAFFTReceiver(
            FFTAdapterConfig(
                frame_bins=4,
                useful_bins=4,
                use_i2s_tags=True,
                require_bfpexp_before_fft=False,
                handshake_timeout_seconds=0.01,
            )
        )
        relaxed_rx._proc = FakeProcess(stream)
        frame = relaxed_rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0, 17.0], dtype=np.float32))

    def test_tagged_mode_with_done_line_can_bootstrap_from_fft_without_bfpexp(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            done_line=24,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (2, 1, 2),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, 8, 15),
                ]
            )
        )

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0, 17.0], dtype=np.float32))

    def test_tagged_mode_discards_idle_pairs_while_searching_for_frame(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, use_i2s_tags=True, handshake_timeout_seconds=0.01)
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (0, 0, 0),
                    (0, 0, 0),
                    (1, 7, 7),
                    (0, 0, 0),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, -8, 15),
                    (2, 7, -24),
                ]
            )
        )

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_tagged_mode_resynchronizes_after_broken_frame(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, use_i2s_tags=True, handshake_timeout_seconds=0.01)
        rx = FPGAFFTReceiver(cfg)
        stream = pack_tagged_pairs(
            [
                (1, 9, 9),
                (2, 1, 2),
                (2, 3, 4),
                (0, 0, 0),
                (1, 5, 5),
                (2, 3, 4),
                (2, 5, 12),
                (2, 8, 15),
                (2, 7, 24),
            ]
        )
        rx._proc = FakeProcess(stream)

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_raw_mode_reads_exact_frame_without_tags(self):
        cfg = FFTAdapterConfig(frame_bins=3, useful_bins=3, use_i2s_tags=False)
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(pack_raw_pairs([(1, 2), (3, 4), (5, 12)]))

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0], dtype=np.float32))

    def test_raw_mode_can_wait_for_gpio_falling_edge_before_consuming_frame(self):
        cfg = FFTAdapterConfig(
            frame_bins=3,
            useful_bins=3,
            use_i2s_tags=False,
            bfpexp_flag_line=23,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_raw_pairs([(100, 100), (200, 200), (1, 2), (3, 4), (5, 12)]),
            max_chunk_bytes=16,
        )
        rx._bfpexp_line = object()
        rx._gpio_api = "v1"
        rx._read_flag_active = mock.Mock(side_effect=[True, False])

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
