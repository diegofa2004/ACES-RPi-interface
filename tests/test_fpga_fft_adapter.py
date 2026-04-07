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
from tests.test_support import FakeProcess, pack_raw_pairs


class FFTAdapterConfigTests(unittest.TestCase):
    def test_rejects_useful_bins_above_rfft_limit(self):
        with self.assertRaises(ValueError):
            FFTAdapterConfig(frame_bins=8, useful_bins=6)

    def test_rejects_unknown_channel_mode(self):
        with self.assertRaises(ValueError):
            FFTAdapterConfig(channel_mode="banana")


class FPGAFFTReceiverTests(unittest.TestCase):
    def test_read_available_pairs_can_return_partial_chunk(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=3)
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(pack_raw_pairs([(1, 2), (3, 4), (5, 6)]), max_chunk_bytes=16)

        pairs = rx.read_available_pairs(3)
        self.assertIsNotNone(pairs)
        assert pairs is not None
        self.assertEqual(pairs.shape, (2, 2))
        np.testing.assert_array_equal(pairs, np.asarray([[1, 2], [3, 4]], dtype=np.int32))

    def test_start_and_stop_use_capture_process_helpers(self):
        fake_proc = FakeProcess(b"")
        cfg = FFTAdapterConfig(device="auto")
        rx = FPGAFFTReceiver(cfg)

        with mock.patch.object(fpga_fft_adapter, "resolve_audio_device", return_value="hw:2,0"), \
             mock.patch.object(fpga_fft_adapter, "build_capture_cmd", return_value=["fake-capture", "--raw"]) as build_mock, \
             mock.patch.object(fpga_fft_adapter, "start_capture_process", return_value=fake_proc) as start_mock, \
             mock.patch.object(fpga_fft_adapter, "stop_process") as stop_mock:
            rx.start()
            build_mock.assert_called_once_with(
                "hw:2,0",
                48828,
                backend="auto",
                capture_binary=None,
            )
            start_mock.assert_called_once_with(
                "hw:2,0",
                48828,
                backend="auto",
                capture_binary=None,
            )
            rx.stop()
            stop_mock.assert_called_once_with(fake_proc)

    def test_read_frame_computes_fft_and_mfcc_from_left_channel(self):
        cfg = FFTAdapterConfig(
            frame_bins=8,
            useful_bins=5,
            channel_mode="left",
            remove_dc=False,
        )
        rx = FPGAFFTReceiver(cfg)
        samples = np.asarray([0, 1000, 0, -1000, 0, 1000, 0, -1000], dtype=np.int32)
        stereo = np.column_stack((samples, np.zeros_like(samples)))
        rx._proc = FakeProcess(stereo.astype(np.int32).tobytes())

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, mfcc = frame

        expected_fft = np.abs(np.fft.rfft(samples.astype(np.float32) * np.hanning(8).astype(np.float32), n=8))
        np.testing.assert_allclose(fft_bins, expected_fft[:5], rtol=1e-5, atol=1e-5)
        self.assertEqual(mfcc.shape, (13,))
        self.assertEqual(rx.last_channel_used, "left")

    def test_auto_channel_mode_picks_the_louder_channel(self):
        cfg = FFTAdapterConfig(
            frame_bins=8,
            useful_bins=5,
            channel_mode="auto",
            remove_dc=False,
        )
        rx = FPGAFFTReceiver(cfg)
        left = np.asarray([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int32)
        right = left * 100
        stereo = np.column_stack((left, right))
        rx._proc = FakeProcess(stereo.astype(np.int32).tobytes())

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame

        expected_fft = np.abs(np.fft.rfft(right.astype(np.float32) * np.hanning(8).astype(np.float32), n=8))
        np.testing.assert_allclose(fft_bins, expected_fft[:5], rtol=1e-5, atol=1e-5)
        self.assertEqual(rx.last_channel_used, "right")

    def test_sample_shift_bits_are_applied_before_fft(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=3,
            channel_mode="left",
            sample_shift_bits=2,
            remove_dc=False,
        )
        rx = FPGAFFTReceiver(cfg)
        left = np.asarray([0, 16, 0, -16], dtype=np.int32)
        stereo = np.column_stack((left, np.zeros_like(left)))
        rx._proc = FakeProcess(stereo.astype(np.int32).tobytes())

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame

        shifted = np.right_shift(left, 2).astype(np.float32)
        expected_fft = np.abs(np.fft.rfft(shifted * np.hanning(4).astype(np.float32), n=4))
        np.testing.assert_allclose(fft_bins, expected_fft[:3], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
