import argparse
import time
from pathlib import Path

try:
    import gpiod  # type: ignore
except ImportError:  # pragma: no cover - optional dependency on target device
    gpiod = None


DEFAULT_TRIGGER_FILE = Path(__file__).resolve().parent / "record_button.trigger"


class GPIOButtonInput:
    def __init__(self, chip_path: str, line_offset: int):
        if gpiod is None:
            raise RuntimeError(
                "GPIO button requested but python gpiod is not installed. "
                "Install python3-libgpiod on the Raspberry Pi."
            )

        self.chip_path = chip_path
        self.line_offset = line_offset
        self._gpio_chip = None
        self._line_request = None
        self._line = None
        self._gpio_api = None

    def open(self) -> None:
        chip = gpiod.Chip(self.chip_path)
        self._gpio_chip = chip

        if hasattr(gpiod, "LineSettings"):
            line_module = getattr(gpiod, "line", gpiod)
            settings = {
                self.line_offset: gpiod.LineSettings(
                    direction=line_module.Direction.INPUT,
                )
            }
            if hasattr(chip, "request_lines"):
                self._line_request = chip.request_lines(
                    consumer="record_button",
                    config=settings,
                )
            else:
                self._line_request = gpiod.request_lines(
                    self.chip_path,
                    consumer="record_button",
                    config=settings,
                )
            self._gpio_api = "v2"
            return

        line = chip.get_line(self.line_offset)
        line.request(consumer="record_button", type=gpiod.LINE_REQ_DIR_IN)
        self._line = line
        self._gpio_api = "v1"

    def read_active(self) -> bool:
        if self._gpio_api == "v1":
            if self._line is None:
                return False
            return bool(self._line.get_value())

        if self._line_request is None:
            return False
        line_module = getattr(gpiod, "line", gpiod)
        value = self._line_request.get_value(self.line_offset)
        return value == line_module.Value.ACTIVE

    def close(self) -> None:
        if self._line_request is not None:
            release = getattr(self._line_request, "release", None)
            if callable(release):
                release()
            self._line_request = None

        if self._line is not None:
            release = getattr(self._line, "release", None)
            if callable(release):
                release()
            self._line = None

        close = getattr(self._gpio_chip, "close", None)
        if callable(close):
            close()
        self._gpio_chip = None
        self._gpio_api = None


def write_trigger(trigger_file: Path) -> None:
    trigger_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = trigger_file.with_suffix(trigger_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="ascii") as handle:
        handle.write(f"{time.time():.6f}\n")
    tmp_file.replace(trigger_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read a Raspberry Pi GPIO button (polling) and trigger event recording in analyzer_from_fpga_fft.py."
    )
    parser.add_argument("--gpio-chip", default="/dev/gpiochip0", help="GPIO chip path")
    parser.add_argument("--button-line", type=int, required=True, help="GPIO line offset connected to the button")
    parser.add_argument(
        "--active-low",
        action="store_true",
        help="Treat the button as pressed when the GPIO input is low",
    )
    parser.add_argument(
        "--trigger-file",
        default=str(DEFAULT_TRIGGER_FILE),
        help="File touched when the button is pressed",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.02, help="GPIO polling interval in seconds")
    parser.add_argument("--debounce-ms", type=float, default=250.0, help="Minimum time between presses")
    args = parser.parse_args()

    if args.button_line < 0:
        parser.error("--button-line must be non-negative")
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if args.debounce_ms < 0.0:
        parser.error("--debounce-ms must be non-negative")

    trigger_file = Path(args.trigger_file).expanduser().resolve()
    button = GPIOButtonInput(args.gpio_chip, args.button_line)
    debounce_seconds = args.debounce_ms / 1000.0

    try:
        button.open()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

    print("GPIO button bridge active.", flush=True)
    print("GPIO chip:", args.gpio_chip, flush=True)
    print("Button line:", args.button_line, flush=True)
    print("Active low:", bool(args.active_low), flush=True)
    print("Trigger file:", trigger_file, flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    last_pressed = False
    last_trigger_time = 0.0

    try:
        while True:
            raw_active = button.read_active()
            pressed = (not raw_active) if args.active_low else raw_active
            now = time.monotonic()

            if pressed and (not last_pressed) and ((now - last_trigger_time) >= debounce_seconds):
                write_trigger(trigger_file)
                last_trigger_time = now
                print("Button press detected: recording trigger sent.", flush=True)

            last_pressed = pressed
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Stopping button bridge...", flush=True)
    finally:
        button.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
