import argparse
import os
import threading
import time
from collections import deque

try:
    from .fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
except ImportError:
    from fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "hw:2,0")


def main() -> int:
    parser = argparse.ArgumentParser(description="Feed analyzer buffers from FPGA I2S FFT stream.")
    parser.add_argument("-D", "--device", default=DEFAULT_AUDIO_DEVICE, help="ALSA capture device")
    parser.add_argument("-r", "--rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--frame-bins", type=int, default=512, help="Complex bins per FPGA FFT frame")
    parser.add_argument("--useful-bins", type=int, default=256, help="Bins kept for similarity")
    parser.add_argument("--gpio-chip", default="/dev/gpiochip0", help="GPIO chip used for handshake")
    parser.add_argument(
        "--bfpexp-flag-line",
        type=int,
        default=None,
        help="Input GPIO line number: active during BFPEXP transmission",
    )
    parser.add_argument(
        "--done-line",
        type=int,
        default=None,
        help="Output GPIO line number: pulsed when 512 FFT bins are consumed",
    )
    parser.add_argument(
        "--flag-active-low",
        action="store_true",
        help="Set when BFPEXP flag signal is active-low instead of active-high",
    )
    parser.add_argument(
        "--wait-low-level",
        action="store_true",
        help="Wait for BFPEXP flag low level instead of requiring a falling edge",
    )
    parser.add_argument(
        "--done-pulse-ms",
        type=float,
        default=0.5,
        help="Done pulse width in milliseconds",
    )
    parser.add_argument(
        "--handshake-timeout-ms",
        type=float,
        default=1000.0,
        help="Timeout waiting for FFT window trigger in milliseconds",
    )
    parser.add_argument(
        "--use-i2s-tags",
        action="store_true",
        help="Decode per-word in-band tags (idle/BFPEXP/FFT) from I2S stream",
    )
    parser.add_argument("--tag-shift", type=int, default=30, help="Bit shift of type tag in each 32-bit word")
    parser.add_argument("--tag-mask", type=lambda v: int(v, 0), default=0x3, help="Bitmask for type tag")
    parser.add_argument("--payload-bits", type=int, default=18, help="Signed payload width inside each word")
    parser.add_argument("--tag-idle", type=int, default=0, help="Tag value representing idle/no data")
    parser.add_argument("--tag-bfpexp", type=int, default=1, help="Tag value representing BFPEXP data")
    parser.add_argument("--tag-fft", type=int, default=2, help="Tag value representing FFT complex bins")
    parser.add_argument(
        "--allow-fft-without-bfpexp",
        action="store_true",
        help="Accept FFT-tagged frame start even if no BFPEXP tag was observed first",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.frame_bins <= 0:
        parser.error("--frame-bins must be positive")
    if not 2 <= args.useful_bins <= args.frame_bins:
        parser.error("--useful-bins must satisfy 2 <= useful-bins <= frame-bins")
    if args.payload_bits <= 0:
        parser.error("--payload-bits must be positive")

    lock = threading.Lock()
    buffer2 = deque(maxlen=330)  # MFCC history (15 s equivalent windowing in original code)
    buffer4 = deque(maxlen=330)  # FFT magnitude history

    try:
        cfg = FFTAdapterConfig(
            device=args.device,
            sample_rate=args.rate,
            frame_bins=args.frame_bins,
            useful_bins=args.useful_bins,
            gpio_chip=args.gpio_chip,
            bfpexp_flag_line=args.bfpexp_flag_line,
            done_line=args.done_line,
            flag_active_high=not args.flag_active_low,
            done_pulse_seconds=max(0.0, args.done_pulse_ms / 1000.0),
            handshake_timeout_seconds=max(0.001, args.handshake_timeout_ms / 1000.0),
            wait_for_flag_falling_edge=not args.wait_low_level,
            use_i2s_tags=args.use_i2s_tags,
            tag_shift=args.tag_shift,
            tag_mask=args.tag_mask,
            payload_bits=args.payload_bits,
            tag_idle=args.tag_idle,
            tag_bfpexp=args.tag_bfpexp,
            tag_fft=args.tag_fft,
            require_bfpexp_before_fft=not args.allow_fft_without_bfpexp,
        )
    except ValueError as exc:
        parser.error(str(exc))

    rx = FPGAFFTReceiver(cfg)
    try:
        rx.start()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

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
