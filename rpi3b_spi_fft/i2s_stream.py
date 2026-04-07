import os
import re
import subprocess
from pathlib import Path
from typing import Optional


AUTO_AUDIO_DEVICE = "auto"
DEFAULT_CAPTURE_BACKEND = "auto"
DEFAULT_CAPTURE_RATE_HZ = 48_828
DEFAULT_CHANNEL_COUNT = 2
DEFAULT_SAMPLE_BYTES = 4
DEFAULT_READ_FRAMES = 512
DEFAULT_CAPTURE_BINARY = str(Path(__file__).resolve().with_name("alsa_logger"))

_ARECORD_CARD_RE = re.compile(r"card\s+(?P<card>\d+):.*device\s+(?P<device>\d+):", re.IGNORECASE)


def normalize_audio_device(device: str) -> str:
    requested = (device or "").strip()
    return requested


def list_audio_devices() -> list[str]:
    try:
        proc = subprocess.run(
            ["arecord", "-l"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    devices: list[str] = []
    for line in proc.stdout.splitlines():
        match = _ARECORD_CARD_RE.search(line)
        if match is None:
            continue
        devices.append(f"hw:{match.group('card')},{match.group('device')}")
    return devices


def resolve_audio_device(device: str) -> str:
    requested = normalize_audio_device(device)
    if requested and requested.lower() != AUTO_AUDIO_DEVICE:
        return requested

    env_device = normalize_audio_device(os.environ.get("AUDIO_DEVICE", ""))
    if env_device:
        return env_device

    devices = list_audio_devices()
    if "hw:2,0" in devices:
        return "hw:2,0"
    if devices:
        return devices[0]
    return "hw:2,0"


def resolve_capture_backend(backend: str, capture_binary: Optional[str]) -> str:
    requested = (backend or DEFAULT_CAPTURE_BACKEND).strip().lower()
    if requested in {"arecord", "alsa-c"}:
        return requested

    binary_path = Path(capture_binary or DEFAULT_CAPTURE_BINARY).expanduser()
    if binary_path.exists():
        return "alsa-c"
    return "arecord"


def build_arecord_command(
    device: str,
    *,
    sample_rate: int,
) -> list[str]:
    return [
        "arecord",
        "-q",
        "-D",
        device,
        "-B",
        "250000",
        "-F",
        "50000",
        "-f",
        "S32_LE",
        "-c",
        str(DEFAULT_CHANNEL_COUNT),
        "-r",
        str(sample_rate),
        "-t",
        "raw",
    ]


def build_alsa_capture_command(
    device: str,
    *,
    sample_rate: int,
    read_frames: int,
    capture_binary: Optional[str],
) -> list[str]:
    binary_path = str(Path(capture_binary or DEFAULT_CAPTURE_BINARY).expanduser())
    return [
        binary_path,
        "--device",
        device,
        "--rate",
        str(sample_rate),
        "--read-frames",
        str(read_frames),
        "--mode",
        "raw",
        "--output",
        "-",
        "--stats-interval-ms",
        "0",
    ]
