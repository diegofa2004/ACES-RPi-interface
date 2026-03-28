import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import numpy as np

from fft_shared import DEFAULT_SHM_NAME, FFTSharedState, STATUS_NO_DATA, STATUS_OK


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "hw:2,0")


def build_arecord_cmd(device: str, rate: int) -> list:
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

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        with open(args.csv, "w", newline="", encoding="ascii") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["timestamp_ns", "seq", "real", "imag", "status"])

            while not stop:
                if proc.stdout is None:
                    raise RuntimeError("arecord stdout pipe is unavailable")

                raw = proc.stdout.read(chunk_bytes)
                if not raw:
                    status = STATUS_NO_DATA
                    state.write(0, 0, seq, status)
                    writer.writerow([time.time_ns(), seq, 0, 0, status])
                    f_csv.flush()

                    if proc.poll() is not None:
                        err = b""
                        if proc.stderr is not None:
                            err = proc.stderr.read()
                        print("arecord exited. stderr:", err.decode(errors="ignore"), file=sys.stderr)
                        return 1

                    seq = (seq + 1) & 0xFFFFFFFF
                    continue

                data = np.frombuffer(raw, dtype=np.int32)
                if data.size < 2:
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
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

        state.close()
        state.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
