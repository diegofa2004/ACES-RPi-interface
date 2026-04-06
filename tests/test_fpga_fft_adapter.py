import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_spi_fft import fpga_fft_adapter
from rpi3b_spi_fft.fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
from tests.test_support import FakeSPI, pack_raw_pairs, pack_tagged_pairs


class FFTAdapterConfigTests(unittest.TestCase):
    def test_rejects_overlapping_tag_and_payload(self):
        with self.assertRaises(ValueError):
            FFTAdapterConfig(tag_shift=10, payload_bits=18)


class FPGAFFTReceiverTests(unittest.TestCase):
    def test_receiver_polls_at_least_one_spi_transaction_per_read(self):
        cfg = FFTAdapterConfig(frame_bins=512, useful_bins=256, bfpexp_hold_frames=1)
        rx = FPGAFFTReceiver(cfg)

        self.assertEqual(rx._transaction_pairs, 513)
        self.assertEqual(rx._poll_pairs, 513)
        self.assertEqual(rx._poll_bytes, 4104)

    def test_read_available_pairs_can_return_partial_chunk_from_buffer(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4)
        rx = FPGAFFTReceiver(cfg)
        rx._byte_buffer.extend(pack_raw_pairs([(1, 2), (3, 4), (5, 6)]))

        pairs = rx.read_available_pairs(2)
        self.assertIsNotNone(pairs)
        assert pairs is not None
        self.assertEqual(pairs.shape, (2, 2))
        np.testing.assert_array_equal(pairs, np.asarray([[1, 2], [3, 4]], dtype=np.int32))

    def test_start_and_stop_use_spi_helpers(self):
        fake_spi = FakeSPI(b"")
        cfg = FFTAdapterConfig(device="auto")
        rx = FPGAFFTReceiver(cfg)

        with mock.patch.object(fpga_fft_adapter, "resolve_spi_device", return_value="/dev/spidev0.0"), \
             mock.patch.object(fpga_fft_adapter, "open_spi_device", return_value=fake_spi) as open_mock, \
             mock.patch.object(fpga_fft_adapter, "close_spi_device") as close_mock:
            rx.start()
            open_mock.assert_called_once_with(
                "/dev/spidev0.0",
                max_speed_hz=cfg.spi_max_speed_hz,
                mode=cfg.spi_mode,
                bits_per_word=cfg.spi_bits_per_word,
            )
            rx.stop()
            close_mock.assert_called_once_with(fake_spi)

    def test_read_frame_in_tagged_mode_decodes_bfpexp_then_fft_pairs(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, handshake_timeout_seconds=0.01)
        rx = FPGAFFTReceiver(cfg)
        rx._byte_buffer.extend(
            pack_tagged_pairs(
                [
                    (1, 7, 7),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, -8, 15),
                    (2, 7, -24),
                ]
            )
        )

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

        strict_rx = FPGAFFTReceiver(FFTAdapterConfig(frame_bins=4, useful_bins=4, handshake_timeout_seconds=0.01))
        strict_rx._byte_buffer.extend(stream)
        self.assertIsNone(strict_rx.read_frame())

        relaxed_rx = FPGAFFTReceiver(
            FFTAdapterConfig(
                frame_bins=4,
                useful_bins=4,
                require_bfpexp_before_fft=False,
                handshake_timeout_seconds=0.01,
            )
        )
        relaxed_rx._byte_buffer.extend(stream)
        frame = relaxed_rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0, 17.0], dtype=np.float32))

    def test_tagged_mode_discards_idle_pairs_while_searching_for_frame(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, handshake_timeout_seconds=0.01)
        rx = FPGAFFTReceiver(cfg)
        rx._byte_buffer.extend(
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
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, handshake_timeout_seconds=0.01)
        rx = FPGAFFTReceiver(cfg)
        rx._byte_buffer.extend(
            pack_tagged_pairs(
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
        )

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_raw_mode_reads_exact_frame_without_tags(self):
        cfg = FFTAdapterConfig(frame_bins=3, useful_bins=3, use_word_tags=False)
        rx = FPGAFFTReceiver(cfg)
        rx._byte_buffer.extend(pack_raw_pairs([(1, 2), (3, 4), (5, 12)]))

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0], dtype=np.float32))

    def test_raw_mode_can_wait_for_window_ready_before_consuming_frame(self):
        cfg = FFTAdapterConfig(
            frame_bins=3,
            useful_bins=3,
            use_word_tags=False,
            window_ready_line=23,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._spi = FakeSPI(pack_raw_pairs([(1, 2), (3, 4), (5, 12)]))
        rx._window_ready_line = object()
        rx._gpio_api = "v1"
        rx._read_window_ready = mock.Mock(side_effect=[False, True])

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
