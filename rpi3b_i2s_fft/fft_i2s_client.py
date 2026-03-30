import argparse
import json
import sys
import time
from typing import Optional

try:
    from .fft_shared import DEFAULT_SHM_NAME, FFTSharedState, STATUS_NO_DATA
except ImportError:
    from fft_shared import DEFAULT_SHM_NAME, FFTSharedState, STATUS_NO_DATA


def open_shared_state(shm_name: str, wait: bool, interval: float) -> Optional[FFTSharedState]:
    warned = False
    while True:
        try:
            return FFTSharedState(name=shm_name, create=False)
        except FileNotFoundError:
            if not wait:
                return None
            if not warned:
                print(
                    f"Shared memory '{shm_name}' not found yet. Waiting for the daemon...",
                    file=sys.stderr,
                    flush=True,
                )
                warned = True
            time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read latest FFT real/imag values from shared memory.")
    parser.add_argument("--shm-name", default=DEFAULT_SHM_NAME, help="Shared memory block name")
    parser.add_argument("--watch", action="store_true", help="Continuously print updates")
    parser.add_argument("--interval", type=float, default=0.05, help="Polling interval seconds for --watch")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive")

    state = open_shared_state(args.shm_name, wait=args.watch, interval=args.interval)
    if state is None:
        print(
            json.dumps(
                {
                    "real": 0,
                    "imag": 0,
                    "seq": 0,
                    "status": STATUS_NO_DATA,
                    "timestamp_ns": time.time_ns(),
                    "error": f"Shared memory '{args.shm_name}' does not exist.",
                }
            ),
            flush=True,
        )
        return 1

    try:
        if args.watch:
            last_seq = None
            while True:
                item = state.read()
                if item["seq"] != last_seq:
                    print(json.dumps(item), flush=True)
                    last_seq = item["seq"]
                time.sleep(args.interval)
        else:
            print(json.dumps(state.read()), flush=True)
    finally:
        state.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
