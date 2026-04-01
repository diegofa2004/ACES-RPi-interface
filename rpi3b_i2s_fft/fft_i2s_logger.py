import argparse
import collections
import csv
import os
import signal
import sys
import time
from typing import Optional

import numpy as np

try:
    from .i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_RATE_HZ,
        TaggedI2SRealigner,
        build_arecord_cmd,
        read_exactly,
        resolve_audio_device,
        start_arecord_process,
        stop_process,
        trim_incomplete_frames,
    )
except ImportError:
    from i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_RATE_HZ,
        TaggedI2SRealigner,
        build_arecord_cmd,
        read_exactly,
        resolve_audio_device,
        start_arecord_process,
        stop_process,
        trim_incomplete_frames,
    )


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE") or AUTO_AUDIO_DEVICE
DEFAULT_LOGGER_CHUNK_FRAMES = 4096
DEFAULT_CSV_FLUSH_EVERY_CHUNKS = 8


def format_i32_hex(value: int) -> str:
    return f"0x{int(value) & 0xFFFFFFFF:08X}"


class MirroredPairNormalizer:
    def __init__(self):
        self._preferred_by_signature: dict[tuple[int, int], tuple[int, int]] = {}

    def normalize(self, stereo: np.ndarray) -> np.ndarray:
        stereo_i32 = np.asarray(stereo, dtype=np.int32)
        if stereo_i32.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        if stereo_i32.ndim != 2 or stereo_i32.shape[1] != 2:
            stereo_i32 = stereo_i32.reshape(-1, 2)

        stereo_u32 = stereo_i32.astype(np.uint32, copy=False)
        counts = collections.Counter((int(left), int(right)) for left, right in stereo_u32)
        chunk_preference: dict[tuple[int, int], tuple[int, int]] = {}

        for (left, right), count in counts.items():
            if left == right:
                continue

            reversed_pair = (right, left)
            key = (left, right) if left < right else (right, left)
            preferred = self._preferred_by_signature.get(key)
            if preferred is None:
                reversed_count = counts.get(reversed_pair, 0)
                if reversed_count > count:
                    preferred = reversed_pair
                else:
                    preferred = (left, right)
                self._preferred_by_signature[key] = preferred
            chunk_preference[key] = preferred

        if not chunk_preference:
            return stereo_i32.copy()

        normalized = stereo_u32.copy()
        for row in normalized:
            left = int(row[0])
            right = int(row[1])
            if left == right:
                continue
            key = (left, right) if left < right else (right, left)
            preferred = chunk_preference.get(key)
            if preferred is None:
                continue
            if (left, right) != preferred:
                row[0], row[1] = row[1], row[0]

        return normalized.view(np.int32)


def decode_stereo_frames(raw: bytes) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.int32)
    if data.size < 2 or (data.size % 2) != 0:
        return np.empty((0, 2), dtype=np.int32)
    return data.reshape(-1, 2)


def write_csv_rows(
    writer: csv.writer,
    stereo: np.ndarray,
    seq_start: int,
    *,
    timestamp_ns_fn=time.time_ns,
) -> int:
    seq = seq_start
    rows = []
    for row in stereo:
        rows.append([timestamp_ns_fn(), seq, format_i32_hex(row[0]), format_i32_hex(row[1])])
        seq = (seq + 1) & 0xFFFFFFFF
    writer.writerows(rows)
    return seq


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture I2S FFT stream and log raw real/imag pairs to CSV."
    )
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_AUDIO_DEVICE,
        help="ALSA capture device (default: $AUDIO_DEVICE if set, otherwise auto-detect)",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=DEFAULT_CAPTURE_RATE_HZ,
        help="Host-side ALSA sample rate in Hz (nominal wire rate is 48828.125 Hz)",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_LOGGER_CHUNK_FRAMES,
        help="Frames read per chunk",
    )
    parser.add_argument("--csv", default="fft_capture.csv", help="Output CSV path")
    parser.add_argument(
        "--flush-every-chunks",
        type=int,
        default=DEFAULT_CSV_FLUSH_EVERY_CHUNKS,
        help="Flush CSV data every N chunks instead of every chunk",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")
    if args.flush_every_chunks <= 0:
        parser.error("--flush-every-chunks must be positive")

    try:
        device = resolve_audio_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    bytes_per_frame = 8  # 2 channels x int32
    chunk_bytes = args.chunk_frames * bytes_per_frame

    seq = 0
    chunk_index = 0
    stop = False
    realigner = TaggedI2SRealigner(preferred_swap_channels=True)
    normalizer = MirroredPairNormalizer()

    def handle_stop(_sig: int, _frame: Optional[object]) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    cmd = build_arecord_cmd(device, args.rate)
    print("Using ALSA capture device:", device, flush=True)
    print("Starting:", " ".join(cmd), flush=True)
    print("Logging CSV to:", args.csv, flush=True)

    try:
        proc = start_arecord_process(device, args.rate)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        with open(args.csv, "w", newline="", encoding="ascii", buffering=1024 * 1024) as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["timestamp_ns", "seq", "real", "imag"])

            while not stop:
                if proc.stdout is None:
                    raise RuntimeError("arecord stdout pipe is unavailable")

                raw = trim_incomplete_frames(read_exactly(proc.stdout, chunk_bytes), bytes_per_frame)
                if not raw:
                    if proc.poll() is not None:
                        print(f"arecord exited with code {proc.returncode}", file=sys.stderr)
                        return 1

                    continue

                stereo = decode_stereo_frames(raw)
                if stereo.size == 0:
                    continue

                stereo = realigner.push_pairs(stereo)
                if stereo.size == 0:
                    continue
                stereo = normalizer.normalize(stereo)

                seq = write_csv_rows(writer, stereo, seq)
                chunk_index += 1
                if (chunk_index % args.flush_every_chunks) == 0:
                    f_csv.flush()

            f_csv.flush()

    finally:
        stop_process(proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
