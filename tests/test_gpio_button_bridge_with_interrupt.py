import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


gpio_button_module = importlib.import_module("rpi3b_i2s_fft.gpio_button_bridge_with_interrupt")


class GPIOButtonInterruptInputTests(unittest.TestCase):
    def test_prefers_rpi_gpio_backend_when_available(self):
        fake_gpio = mock.Mock()
        fake_gpio.BCM = object()
        fake_gpio.IN = object()
        fake_gpio.BOTH = object()
        fake_gpio.input.return_value = 1

        with mock.patch.object(gpio_button_module, "rpi_gpio", fake_gpio), mock.patch.object(
            gpio_button_module, "gpiod", None
        ):
            button = gpio_button_module.GPIOButtonInterruptInput("/dev/gpiochip0", 17)
            button.open()

            self.assertEqual(button._gpio_api, "rpi_gpio")
            fake_gpio.setmode.assert_called_once_with(fake_gpio.BCM)
            fake_gpio.setup.assert_called_once_with(17, fake_gpio.IN)
            fake_gpio.add_event_detect.assert_called_once()
            self.assertTrue(button.read_active())

            button._on_rpi_gpio_edge(17)
            self.assertTrue(button.wait_for_edge(0.0))
            self.assertEqual(button.read_edge_events(), 1)

            button.close()
            fake_gpio.remove_event_detect.assert_called_once_with(17)
            fake_gpio.cleanup.assert_called_once_with(17)


if __name__ == "__main__":
    unittest.main()
