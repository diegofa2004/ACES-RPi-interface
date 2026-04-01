import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import i2s_stream


class I2SStreamTests(unittest.TestCase):
    def test_build_arecord_cmd_uses_expected_format(self):
        cmd = i2s_stream.build_arecord_cmd("hw:1,0", i2s_stream.DEFAULT_CAPTURE_RATE_HZ)
        self.assertEqual(
            cmd,
            [
                "arecord",
                "-q",
                "-D",
                "hw:1,0",
                "-B",
                str(i2s_stream.DEFAULT_ARECORD_BUFFER_TIME_US),
                "-F",
                str(i2s_stream.DEFAULT_ARECORD_PERIOD_TIME_US),
                "-f",
                "S32_LE",
                "-c",
                "2",
                "-r",
                str(i2s_stream.DEFAULT_CAPTURE_RATE_HZ),
                "-t",
                "raw",
            ],
        )

    def test_trim_incomplete_frames_discards_partial_tail(self):
        raw = b"\x00" * 19
        trimmed = i2s_stream.trim_incomplete_frames(raw, bytes_per_frame=8)
        self.assertEqual(len(trimmed), 16)

    def test_resolve_audio_device_prefers_keyword_match(self):
        proc = SimpleNamespace(
            stdout=(
                "card 0: Loopback [Loopback], device 0: Loopback PCM [Loopback PCM]\n"
                "card 2: acesfpgafft [aces-fpgafft], device 0: FPGA FFT Capture [FPGA FFT Capture]\n"
            )
        )
        with mock.patch.object(i2s_stream.subprocess, "run", return_value=proc):
            self.assertEqual(i2s_stream.resolve_audio_device("auto"), "hw:2,0")

    def test_resolve_audio_device_requires_explicit_choice_when_multiple_devices_match_nothing(self):
        proc = SimpleNamespace(
            stdout=(
                "card 0: Loopback [Loopback], device 0: Loopback PCM [Loopback PCM]\n"
                "card 1: USB [USB Audio], device 0: USB Audio [USB Audio]\n"
            )
        )
        with mock.patch.object(i2s_stream.subprocess, "run", return_value=proc):
            with self.assertRaises(RuntimeError) as ctx:
                i2s_stream.resolve_audio_device("auto")
        self.assertIn("Multiple ALSA capture devices were found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
