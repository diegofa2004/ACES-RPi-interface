import argparse
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
        build_arecord_cmd,
        read_exactly,
        resolve_audio_device,
        start_arecord_process,
        stop_process,
        trim_incomplete_frames,
    )


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE") or AUTO_AUDIO_DEVICE


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
    for row in stereo:
        writer.writerow([timestamp_ns_fn(), seq, int(row[0]), int(row[1])])
        seq = (seq + 1) & 0xFFFFFFFF
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
    parser.add_argument("-r", "--rate", type=int, default=48000, help="Sample rate in Hz")
    parser.add_argument("--chunk-frames", type=int, default=256, help="Frames read per chunk")
    parser.add_argument("--csv", default="fft_capture.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")

    try:
        device = resolve_audio_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    bytes_per_frame = 8  # 2 channels x int32
    chunk_bytes = args.chunk_frames * bytes_per_frame

    seq = 0
    stop = False

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
        with open(args.csv, "w", newline="", encoding="ascii") as f_csv:
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

                seq = write_csv_rows(writer, stereo, seq)
                f_csv.flush()

    finally:
        stop_process(proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
