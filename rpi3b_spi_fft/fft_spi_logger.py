import argparse
import csv
import os
import signal
import sys
import time
from typing import Optional

import numpy as np

try:
    from .fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
    from .spi_stream import AUTO_SPI_DEVICE, DEFAULT_SPI_MAX_SPEED_HZ, DEFAULT_SPI_MODE, resolve_spi_device
except ImportError:
    from fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
    from spi_stream import AUTO_SPI_DEVICE, DEFAULT_SPI_MAX_SPEED_HZ, DEFAULT_SPI_MODE, resolve_spi_device


DEFAULT_SPI_DEVICE = os.environ.get("SPI_DEVICE") or AUTO_SPI_DEVICE


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
        description="Capture the raw tagged SPI FFT payload and log real/imag pairs to CSV."
    )
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_SPI_DEVICE,
        help="SPI device (default: $SPI_DEVICE if set, otherwise auto-detect)",
    )
    parser.add_argument("--spi-max-speed-hz", type=int, default=DEFAULT_SPI_MAX_SPEED_HZ, help="SPI clock rate")
    parser.add_argument("--spi-mode", type=int, default=DEFAULT_SPI_MODE, help="SPI mode")
    parser.add_argument("--frame-bins", type=int, default=512, help="FFT bins per transaction")
    parser.add_argument("--bfpexp-hold-frames", type=int, default=1, help="Tagged BFPEXP pairs per transaction")
    parser.add_argument("--window-ready-line", type=int, default=None, help="GPIO input line used as window_ready")
    parser.add_argument("--chunk-frames", type=int, default=256, help="Pairs emitted to CSV per flush")
    parser.add_argument("--csv", default="fft_capture.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.spi_max_speed_hz <= 0:
        parser.error("--spi-max-speed-hz must be positive")
    if args.spi_mode < 0 or args.spi_mode > 3:
        parser.error("--spi-mode must be between 0 and 3")
    if args.frame_bins < 2:
        parser.error("--frame-bins must be at least 2")
    if args.bfpexp_hold_frames <= 0:
        parser.error("--bfpexp-hold-frames must be positive")
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")

    try:
        device = resolve_spi_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    cfg = FFTAdapterConfig(
        device=device,
        sample_rate=48000,
        frame_bins=args.frame_bins,
        useful_bins=args.frame_bins,
        spi_max_speed_hz=args.spi_max_speed_hz,
        spi_mode=args.spi_mode,
        window_ready_line=args.window_ready_line,
        bfpexp_hold_frames=args.bfpexp_hold_frames,
        use_word_tags=True,
    )

    rx = FPGAFFTReceiver(cfg)

    seq = 0
    stop = False

    def handle_stop(_sig: int, _frame: Optional[object]) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print("Using SPI device:", device, flush=True)
    print(
        "Starting SPI raw logger:",
        f"mode={args.spi_mode}",
        f"max_speed_hz={args.spi_max_speed_hz}",
        flush=True,
    )
    print("Logging CSV to:", args.csv, flush=True)

    try:
        rx.start()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        with open(args.csv, "w", newline="", encoding="ascii") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["timestamp_ns", "seq", "real", "imag"])

            while not stop:
                stereo = rx.read_available_pairs(args.chunk_frames)
                if stereo is None or stereo.size == 0:
                    continue

                seq = write_csv_rows(writer, stereo, seq)
                f_csv.flush()

    finally:
        rx.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
