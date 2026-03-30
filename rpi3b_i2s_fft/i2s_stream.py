import subprocess
from typing import BinaryIO


BYTES_PER_STEREO_FRAME = 8


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

