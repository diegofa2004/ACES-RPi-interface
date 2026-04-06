import sys
import unittest
from pathlib import Path
from unittest import mock


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_spi_fft import spi_stream


class SPIStreamTests(unittest.TestCase):
    def test_normalize_spi_device_accepts_short_forms(self):
        self.assertEqual(spi_stream.normalize_spi_device("spidev0.0"), "/dev/spidev0.0")
        self.assertEqual(spi_stream.normalize_spi_device("0.1"), "/dev/spidev0.1")
        self.assertEqual(spi_stream.normalize_spi_device("/dev/spidev1.0"), "/dev/spidev1.0")

    def test_trim_incomplete_frames_discards_partial_tail(self):
        raw = b"\x00" * 19
        trimmed = spi_stream.trim_incomplete_frames(raw, bytes_per_frame=8)
        self.assertEqual(len(trimmed), 16)

    def test_resolve_spi_device_prefers_single_detected_device(self):
        with mock.patch.object(spi_stream, "list_spi_devices", return_value=["/dev/spidev1.1"]):
            self.assertEqual(spi_stream.resolve_spi_device("auto"), "/dev/spidev1.1")

    def test_resolve_spi_device_prefers_spidev0_0_when_multiple_exist(self):
        with mock.patch.object(
            spi_stream,
            "list_spi_devices",
            return_value=["/dev/spidev0.1", "/dev/spidev0.0", "/dev/spidev1.0"],
        ):
            self.assertEqual(spi_stream.resolve_spi_device("auto"), "/dev/spidev0.0")

    def test_resolve_spi_device_requires_explicit_choice_when_multiple_devices_match_nothing(self):
        with mock.patch.object(
            spi_stream,
            "list_spi_devices",
            return_value=["/dev/spidev1.2", "/dev/spidev2.3"],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                spi_stream.resolve_spi_device("auto")
        self.assertIn("Multiple SPI devices were found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
