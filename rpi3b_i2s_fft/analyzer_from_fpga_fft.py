import argparse
import threading
import time
from collections import deque

from fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver


def main() -> int:
    parser = argparse.ArgumentParser(description="Feed analyzer buffers from FPGA I2S FFT stream.")
    parser.add_argument("-D", "--device", default="hw:0,0", help="ALSA capture device")
    parser.add_argument("-r", "--rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--frame-bins", type=int, default=512, help="Complex bins per FPGA FFT frame")
    parser.add_argument("--useful-bins", type=int, default=256, help="Bins kept for similarity")
    args = parser.parse_args()

    lock = threading.Lock()
    buffer2 = deque(maxlen=330)  # MFCC history (15 s equivalent windowing in original code)
    buffer4 = deque(maxlen=330)  # FFT magnitude history

    cfg = FFTAdapterConfig(
        device=args.device,
        sample_rate=args.rate,
        frame_bins=args.frame_bins,
        useful_bins=args.useful_bins,
    )
    rx = FPGAFFTReceiver(cfg)
    rx.start()

    print("Reading FPGA FFT stream from I2S...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            frame = rx.read_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            fft_bins, mfcc = frame
            with lock:
                buffer2.append(mfcc[:8])
                buffer4.append(fft_bins)

            # Replace this print with your analyzer callback if desired.
            print(
                f"frame ok | mfcc0={float(mfcc[0]):.3f} mfcc1={float(mfcc[1]):.3f} "
                f"fft_peak={float(fft_bins.max()):.3f}",
                flush=True,
            )

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        rx.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
