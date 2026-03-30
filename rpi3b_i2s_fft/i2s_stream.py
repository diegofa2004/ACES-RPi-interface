import os
import re
import subprocess
from typing import BinaryIO


AUTO_AUDIO_DEVICE = "auto"
BYTES_PER_STEREO_FRAME = 8
_CAPTURE_DEVICE_RE = re.compile(r"^card\s+(?P<card>\d+):.*device\s+(?P<device>\d+):", re.IGNORECASE)
_PREFERRED_CAPTURE_KEYWORDS = (
    "googlevoicehat",
    "voicehat",
    "voice hat",
    "aiy",
    "i2s",
    "snd_rpi",
    "sndrpi",
)


def build_arecord_cmd(device: str, rate: int) -> list[str]:
    return [
        "arecord",
        "-q",
        "-D",
        device,
        "-f",
        "S32_LE",
        "-c",
        "2",
        "-r",
        str(rate),
        "-t",
        "raw",
    ]


def start_arecord_process(device: str, rate: int) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            build_arecord_cmd(device, rate),
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'arecord' command was not found. Install alsa-utils on the Raspberry Pi."
        ) from exc


def list_capture_devices() -> list[tuple[str, str]]:
    try:
        proc = subprocess.run(
            ["arecord", "-l"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'arecord' command was not found. Install alsa-utils on the Raspberry Pi."
        ) from exc

    devices: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        match = _CAPTURE_DEVICE_RE.match(line.strip())
        if not match:
            continue
        hw_name = f"hw:{match.group('card')},{match.group('device')}"
        devices.append((hw_name, line.strip()))
    return devices


def resolve_audio_device(device: str) -> str:
    requested = (device or "").strip()
    if requested and requested.lower() != AUTO_AUDIO_DEVICE:
        return requested

    env_device = os.environ.get("AUDIO_DEVICE", "").strip()
    if env_device:
        return env_device

    devices = list_capture_devices()
    if not devices:
        raise RuntimeError(
            "No ALSA capture device was found. Check the dtoverlay, reboot the Pi, and run 'arecord -l'."
        )

    if len(devices) == 1:
        return devices[0][0]

    for keyword in _PREFERRED_CAPTURE_KEYWORDS:
        for hw_name, description in devices:
            if keyword in description.lower():
                return hw_name

    found = ", ".join(f"{hw_name} ({description})" for hw_name, description in devices[:4])
    raise RuntimeError(
        "Multiple ALSA capture devices were found. "
        f"Run 'arecord -l' and pass '--device hw:X,Y'. Detected: {found}"
    )


def read_exactly(stream: BinaryIO, byte_count: int) -> bytes:
    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")

    chunks = bytearray()
    while len(chunks) < byte_count:
        piece = stream.read(byte_count - len(chunks))
        if not piece:
            break
        chunks.extend(piece)
    return bytes(chunks)


def trim_incomplete_frames(raw: bytes, bytes_per_frame: int = BYTES_PER_STEREO_FRAME) -> bytes:
    if bytes_per_frame <= 0:
        raise ValueError("bytes_per_frame must be positive")
    valid_size = len(raw) - (len(raw) % bytes_per_frame)
    return raw[:valid_size]


def stop_process(proc: subprocess.Popen, timeout: float = 2.0) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
