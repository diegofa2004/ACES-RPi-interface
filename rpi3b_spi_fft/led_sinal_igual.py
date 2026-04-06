import argparse
import time
from pathlib import Path

try:
    import gpiod  # type: ignore
except ImportError:  # pragma: no cover - optional dependency on target device
    gpiod = None


DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "similaridade.flag"


class GPIOLedOutput:
    def __init__(self, chip_path: str, line_offset: int, active_low: bool):
        if gpiod is None:
            raise RuntimeError(
                "GPIO LED requested but python gpiod is not installed. "
                "Install python3-libgpiod on the Raspberry Pi."
            )

        self.chip_path = chip_path
        self.line_offset = line_offset
        self.active_low = active_low
        self._gpio_chip = None
        self._line_request = None
        self._line = None
        self._gpio_api = None

    def open(self) -> None:
        chip = gpiod.Chip(self.chip_path)
        self._gpio_chip = chip

        if hasattr(gpiod, "LineSettings"):
            line_module = getattr(gpiod, "line", gpiod)
            inactive_value = line_module.Value.ACTIVE if self.active_low else line_module.Value.INACTIVE
            settings = {
                self.line_offset: gpiod.LineSettings(
                    direction=line_module.Direction.OUTPUT,
                    output_value=inactive_value,
                )
            }
            if hasattr(chip, "request_lines"):
                self._line_request = chip.request_lines(
                    consumer="similarity_led",
                    config=settings,
                )
            else:
                self._line_request = gpiod.request_lines(
                    self.chip_path,
                    consumer="similarity_led",
                    config=settings,
                )
            self._gpio_api = "v2"
            self.set_active(False)
            return

        line = chip.get_line(self.line_offset)
        line.request(consumer="similarity_led", type=gpiod.LINE_REQ_DIR_OUT)
        self._line = line
        self._gpio_api = "v1"
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        output_high = not active if self.active_low else active

        if self._gpio_api == "v1":
            if self._line is not None:
                self._line.set_value(1 if output_high else 0)
            return

        if self._line_request is None:
            return
        line_module = getattr(gpiod, "line", gpiod)
        value = line_module.Value.ACTIVE if output_high else line_module.Value.INACTIVE
        self._line_request.set_value(self.line_offset, value)

    def close(self) -> None:
        self.set_active(False)

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


def read_similarity_state(state_file: Path) -> bool:
    try:
        contents = state_file.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return False
    return contents == "1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive a Raspberry Pi LED from compararEvento similarity detections."
    )
    parser.add_argument("--gpio-chip", default="/dev/gpiochip0", help="GPIO chip path")
    parser.add_argument("--led-line", type=int, required=True, help="GPIO line offset connected to the LED")
    parser.add_argument(
        "--active-low",
        action="store_true",
        help="Use this when the LED turns on with a low GPIO level",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help="File written by compararEvento with 0/1 similarity state",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.05, help="File polling interval in seconds")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Keep the LED on for at least this many seconds after the last detection",
    )
    args = parser.parse_args()

    if args.led_line < 0:
        parser.error("--led-line must be non-negative")
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds must be non-negative")

    state_file = Path(args.state_file).expanduser().resolve()
    led = GPIOLedOutput(args.gpio_chip, args.led_line, args.active_low)

    try:
        led.open()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

    print("Similarity LED monitor active.", flush=True)
    print("GPIO chip:", args.gpio_chip, flush=True)
    print("LED line:", args.led_line, flush=True)
    print("Active low:", bool(args.active_low), flush=True)
    print("State file:", state_file, flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    led_is_on = False
    last_detect_time = 0.0

    try:
        while True:
            is_detected = read_similarity_state(state_file)
            now = time.monotonic()

            if is_detected:
                last_detect_time = now

            should_turn_on = is_detected or ((now - last_detect_time) < args.hold_seconds)
            if should_turn_on != led_is_on:
                led.set_active(should_turn_on)
                led_is_on = should_turn_on
                print(f"LED={'ON' if led_is_on else 'OFF'}", flush=True)

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Stopping LED monitor...", flush=True)
    finally:
        led.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
