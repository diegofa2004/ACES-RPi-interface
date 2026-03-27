import argparse
import signal
import subprocess
import sys
from typing import Optional

import numpy as np

from fft_shared import DEFAULT_SHM_NAME, FFTSharedState, STATUS_NO_DATA, STATUS_OK


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
    parser = argparse.ArgumentParser(description="Read I2S stereo stream and publish latest real/imag via shared memory.")
    parser.add_argument("-D", "--device", default="hw:0,0", help="ALSA capture device (default: hw:0,0)")
    parser.add_argument("-r", "--rate", type=int, default=48000, help="Sample rate in Hz")
    parser.add_argument("--chunk-frames", type=int, default=256, help="Frames read per chunk")
    parser.add_argument("--shm-name", default=DEFAULT_SHM_NAME, help="Shared memory block name")
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

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        while not stop:
            if proc.stdout is None:
                raise RuntimeError("arecord stdout pipe is unavailable")

            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                state.write(0, 0, seq, STATUS_NO_DATA)
                if proc.poll() is not None:
                    err = b""
                    if proc.stderr is not None:
                        err = proc.stderr.read()
                    print("arecord exited. stderr:", err.decode(errors="ignore"), file=sys.stderr)
                    return 1
                continue

            data = np.frombuffer(raw, dtype=np.int32)
            if data.size < 2:
                continue

            stereo = data.reshape(-1, 2)
            last = stereo[-1]
            real = int(last[0])
            imag = int(last[1])

            state.write(real, imag, seq, STATUS_OK)
            seq = (seq + 1) & 0xFFFFFFFF

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
