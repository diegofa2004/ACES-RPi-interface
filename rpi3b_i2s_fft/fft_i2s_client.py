import argparse
import json
import time

from fft_shared import DEFAULT_SHM_NAME, FFTSharedState


def main() -> int:
    parser = argparse.ArgumentParser(description="Read latest FFT real/imag values from shared memory.")
    parser.add_argument("--shm-name", default=DEFAULT_SHM_NAME, help="Shared memory block name")
    parser.add_argument("--watch", action="store_true", help="Continuously print updates")
    parser.add_argument("--interval", type=float, default=0.05, help="Polling interval seconds for --watch")
    args = parser.parse_args()

    state = FFTSharedState(name=args.shm_name, create=False)
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
