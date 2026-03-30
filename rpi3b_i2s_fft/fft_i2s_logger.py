import argparse
import csv
import os
import signal
import sys
import time
from typing import Optional

import numpy as np

try:
    from .fft_shared import DEFAULT_SHM_NAME, FFTSharedState, STATUS_NO_DATA, STATUS_OK
    from .i2s_stream import (
        build_arecord_cmd,
        read_exactly,
        start_arecord_process,
        stop_process,
        trim_incomplete_frames,
    )
except ImportError:
    from fft_shared import DEFAULT_SHM_NAME, FFTSharedState, STATUS_NO_DATA, STATUS_OK
    from i2s_stream import (
        build_arecord_cmd,
        read_exactly,
        start_arecord_process,
        stop_process,
        trim_incomplete_frames,
    )


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "hw:2,0")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture I2S FFT stream, log to CSV, and publish latest real/imag to shared memory."
    )
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_AUDIO_DEVICE,
        help=f"ALSA capture device (default: {DEFAULT_AUDIO_DEVICE})",
    )
    parser.add_argument("-r", "--rate", type=int, default=48000, help="Sample rate in Hz")
    parser.add_argument("--chunk-frames", type=int, default=256, help="Frames read per chunk")
    parser.add_argument("--shm-name", default=DEFAULT_SHM_NAME, help="Shared memory block name")
    parser.add_argument("--csv", default="fft_capture.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")

    bytes_per_frame = 8  # 2 channels x int32
    chunk_bytes = args.chunk_frames * bytes_per_frame

    state = FFTSharedState(name=args.shm_name, create=True)
    seq = 0
    stop = False

    def handle_stop(_sig: int, _frame: Optional[object]) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    cmd = build_arecord_cmd(args.device, args.rate)
    print("Starting:", " ".join(cmd), flush=True)
    print("Logging CSV to:", args.csv, flush=True)

    try:
        proc = start_arecord_process(args.device, args.rate)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        state.close()
        state.unlink()
        return 1

    try:
        with open(args.csv, "w", newline="", encoding="ascii") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["timestamp_ns", "seq", "real", "imag", "status"])

            while not stop:
                if proc.stdout is None:
                    raise RuntimeError("arecord stdout pipe is unavailable")

                raw = trim_incomplete_frames(read_exactly(proc.stdout, chunk_bytes), bytes_per_frame)
                if not raw:
                    status = STATUS_NO_DATA
                    state.write(0, 0, seq, status)
                    writer.writerow([time.time_ns(), seq, 0, 0, status])
                    f_csv.flush()

                    if proc.poll() is not None:
                        print(f"arecord exited with code {proc.returncode}", file=sys.stderr)
                        return 1

                    seq = (seq + 1) & 0xFFFFFFFF
                    continue

                data = np.frombuffer(raw, dtype=np.int32)
                if data.size < 2 or (data.size % 2) != 0:
                    continue

                stereo = data.reshape(-1, 2)
                ts_ns = time.time_ns()

                for row in stereo:
                    real = int(row[0])
                    imag = int(row[1])
                    status = STATUS_OK

                    state.write(real, imag, seq, status)
                    writer.writerow([ts_ns, seq, real, imag, status])
                    seq = (seq + 1) & 0xFFFFFFFF

                f_csv.flush()

    finally:
        stop_process(proc)

        state.close()
        state.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
