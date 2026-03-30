import argparse
import os
import signal
import sys
from typing import Optional

import numpy as np

try:
    from .fft_shared import DEFAULT_SHM_NAME, FFTSharedState, STATUS_NO_DATA, STATUS_OK
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
    from fft_shared import DEFAULT_SHM_NAME, FFTSharedState, STATUS_NO_DATA, STATUS_OK
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read I2S stereo stream and publish latest real/imag via shared memory."
    )
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_AUDIO_DEVICE,
        help="ALSA capture device (default: $AUDIO_DEVICE if set, otherwise auto-detect)",
    )
    parser.add_argument("-r", "--rate", type=int, default=48000, help="Sample rate in Hz")
    parser.add_argument("--chunk-frames", type=int, default=256, help="Frames read per chunk")
    parser.add_argument("--shm-name", default=DEFAULT_SHM_NAME, help="Shared memory block name")
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

    state = FFTSharedState(name=args.shm_name, create=True)
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

    try:
        proc = start_arecord_process(device, args.rate)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        state.close()
        state.unlink()
        return 1

    try:
        while not stop:
            if proc.stdout is None:
                raise RuntimeError("arecord stdout pipe is unavailable")

            raw = trim_incomplete_frames(read_exactly(proc.stdout, chunk_bytes), bytes_per_frame)
            if not raw:
                state.write(0, 0, seq, STATUS_NO_DATA)
                seq = (seq + 1) & 0xFFFFFFFF
                if proc.poll() is not None:
                    print(f"arecord exited with code {proc.returncode}", file=sys.stderr)
                    return 1
                continue

            data = np.frombuffer(raw, dtype=np.int32)
            if data.size < 2 or (data.size % 2) != 0:
                continue

            stereo = data.reshape(-1, 2)
            last = stereo[-1]
            real = int(last[0])
            imag = int(last[1])

            state.write(real, imag, seq, STATUS_OK)
            seq = (seq + 1) & 0xFFFFFFFF

    finally:
        stop_process(proc)
        state.close()
        state.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
