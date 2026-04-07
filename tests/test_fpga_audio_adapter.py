import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_spi_fft.fpga_audio_adapter import AudioCaptureConfig, FPGAAudioReceiver
from tests.test_support import FakeProcess


def pack_stereo_frames(left: np.ndarray, right: np.ndarray, *, slot_shift_bits: int) -> bytes:
    stereo = np.stack((left, right), axis=1).astype(np.int32)
    if slot_shift_bits > 0:
        stereo = np.left_shift(stereo, slot_shift_bits).astype(np.int32)
    return stereo.astype("<i4").tobytes()


class AudioCaptureConfigTests(unittest.TestCase):
    def test_rejects_invalid_useful_bins(self):
        with self.assertRaises(ValueError):
            AudioCaptureConfig(frame_length=512, useful_bins=300)


class FPGAAudioReceiverTests(unittest.TestCase):
    def test_read_pcm_frame_auto_selects_active_channel(self):
        cfg = AudioCaptureConfig(
            frame_length=8,
            useful_bins=5,
            sample_shift_bits=8,
            mono_channel="auto",
        )
        rx = FPGAAudioReceiver(cfg)
        left = np.zeros(8, dtype=np.int32)
        right = np.arange(8, dtype=np.int32) * 1000
        rx._proc = FakeProcess(pack_stereo_frames(left, right, slot_shift_bits=8))

        pcm = rx.read_pcm_frame()

        self.assertIsNotNone(pcm)
        assert pcm is not None
        np.testing.assert_array_equal(pcm.astype(np.int32), right)

    def test_read_pcm_frame_can_average_channels(self):
        cfg = AudioCaptureConfig(
            frame_length=4,
            useful_bins=3,
            sample_shift_bits=8,
            mono_channel="average",
        )
        rx = FPGAAudioReceiver(cfg)
        left = np.asarray([100, 200, 300, 400], dtype=np.int32)
        right = np.asarray([300, 200, 100, 0], dtype=np.int32)
        rx._proc = FakeProcess(pack_stereo_frames(left, right, slot_shift_bits=8))

        pcm = rx.read_pcm_frame()

        self.assertIsNotNone(pcm)
        assert pcm is not None
        np.testing.assert_array_equal(pcm.astype(np.int32), np.asarray([200, 200, 200, 200], dtype=np.int32))

    def test_read_frame_computes_fft_and_mfcc_from_pcm(self):
        frame_length = 512
        target_bin = 12
        sample_index = np.arange(frame_length, dtype=np.float32)
        sine = 40_000.0 * np.sin((2.0 * np.pi * target_bin * sample_index) / frame_length)
        pcm = np.asarray(np.round(sine), dtype=np.int32)

        cfg = AudioCaptureConfig(
            frame_length=frame_length,
            useful_bins=256,
            sample_shift_bits=8,
            mono_channel="left",
        )
        rx = FPGAAudioReceiver(cfg)
        rx._proc = FakeProcess(pack_stereo_frames(pcm, np.zeros_like(pcm), slot_shift_bits=8))

        frame = rx.read_frame()

        self.assertIsNotNone(frame)
        fft_bins, mfcc = frame
        self.assertEqual(fft_bins.shape, (256,))
        self.assertEqual(mfcc.shape, (13,))
        self.assertEqual(int(np.argmax(fft_bins)), target_bin)
        self.assertGreater(float(fft_bins[target_bin]), float(np.mean(fft_bins)) * 8.0)


if __name__ == "__main__":
    unittest.main()
