import glob
import os
import re
from typing import Any


AUTO_SPI_DEVICE = "auto"
BYTES_PER_FFT_PAIR = 8
DEFAULT_SPI_MAX_SPEED_HZ = 8_000_000
DEFAULT_SPI_MODE = 0
DEFAULT_SPI_BITS_PER_WORD = 8
_SPIDEV_RE = re.compile(r"spidev(?P<bus>\d+)\.(?P<chip_select>\d+)$")


def normalize_spi_device(device: str) -> str:
    requested = (device or "").strip()
    if not requested:
        return requested
    if requested.startswith("/dev/"):
        return requested
    if requested.startswith("spidev"):
        return f"/dev/{requested}"
    if re.fullmatch(r"\d+\.\d+", requested):
        return f"/dev/spidev{requested}"
    return requested


def list_spi_devices() -> list[str]:
    return sorted(glob.glob("/dev/spidev*"))


def resolve_spi_device(device: str) -> str:
    requested = normalize_spi_device(device)
    if requested and requested.lower() != AUTO_SPI_DEVICE:
        return requested

    env_device = normalize_spi_device(os.environ.get("SPI_DEVICE", ""))
    if env_device:
        return env_device

    devices = list_spi_devices()
    if not devices:
        raise RuntimeError(
            "No SPI device was found. Enable spidev on the Raspberry Pi and check '/dev/spidev*'."
        )

    if len(devices) == 1:
        return devices[0]

    if "/dev/spidev0.0" in devices:
        return "/dev/spidev0.0"

    found = ", ".join(devices[:6])
    raise RuntimeError(
        "Multiple SPI devices were found. Pass '--device /dev/spidevX.Y'. "
        f"Detected: {found}"
    )


def parse_spi_device(device: str) -> tuple[int, int]:
    normalized = normalize_spi_device(device)
    match = _SPIDEV_RE.search(normalized)
    if match is None:
        raise ValueError(
            f"Invalid SPI device '{device}'. Use '/dev/spidevX.Y', 'spidevX.Y', or 'X.Y'."
        )
    return int(match.group("bus")), int(match.group("chip_select"))


def open_spi_device(
    device: str,
    *,
    max_speed_hz: int = DEFAULT_SPI_MAX_SPEED_HZ,
    mode: int = DEFAULT_SPI_MODE,
    bits_per_word: int = DEFAULT_SPI_BITS_PER_WORD,
) -> Any:
    try:
        import spidev  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency on target device
        raise RuntimeError(
            "python spidev is not installed. Install python3-spidev on the Raspberry Pi."
        ) from exc

    bus, chip_select = parse_spi_device(device)
    spi = spidev.SpiDev()
    spi.open(bus, chip_select)
    spi.max_speed_hz = int(max_speed_hz)
    spi.mode = int(mode)
    if hasattr(spi, "bits_per_word"):
        spi.bits_per_word = int(bits_per_word)
    if hasattr(spi, "lsbfirst"):
        spi.lsbfirst = False
    return spi


def close_spi_device(spi: Any) -> None:
    close = getattr(spi, "close", None)
    if callable(close):
        close()


def transfer_exactly(spi: Any, byte_count: int) -> bytes:
    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")
    if byte_count == 0:
        return b""

    tx_data = [0] * byte_count

    if hasattr(spi, "xfer3"):
        rx_data = spi.xfer3(tx_data)
    elif hasattr(spi, "xfer2"):
        rx_data = spi.xfer2(tx_data)
    elif hasattr(spi, "readbytes"):
        rx_data = spi.readbytes(byte_count)
    else:
        raise RuntimeError("Unsupported SPI object: expected xfer3, xfer2, or readbytes support.")

    return bytes(int(value) & 0xFF for value in rx_data)


def trim_incomplete_frames(raw: bytes, bytes_per_frame: int = BYTES_PER_FFT_PAIR) -> bytes:
    if bytes_per_frame <= 0:
        raise ValueError("bytes_per_frame must be positive")
    valid_size = len(raw) - (len(raw) % bytes_per_frame)
    return raw[:valid_size]
