import argparse
import csv
import os
import signal
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from .fpga_fft_adapter import (
        CHANNEL_MODE_AUTO,
        CHANNEL_MODE_AVERAGE,
        CHANNEL_MODE_LEFT,
        CHANNEL_MODE_RIGHT,
        decode_stereo_frames,
        select_mono_channel,
    )
    from .i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_BACKEND,
        DEFAULT_CAPTURE_RATE_HZ,
        build_capture_cmd,
        resolve_audio_device,
        start_capture_process,
        stop_process,
        trim_incomplete_frames,
    )
except ImportError:
    from fpga_fft_adapter import (
        CHANNEL_MODE_AUTO,
        CHANNEL_MODE_AVERAGE,
        CHANNEL_MODE_LEFT,
        CHANNEL_MODE_RIGHT,
        decode_stereo_frames,
        select_mono_channel,
    )
    from i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_BACKEND,
        DEFAULT_CAPTURE_RATE_HZ,
        build_capture_cmd,
        resolve_audio_device,
        start_capture_process,
        stop_process,
        trim_incomplete_frames,
    )


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE") or AUTO_AUDIO_DEVICE
DEFAULT_LOGGER_CHUNK_FRAMES = 4096
DEFAULT_CSV_FLUSH_EVERY_CHUNKS = 8


def format_i32_hex(value: int) -> str:
    return f"0x{int(value) & 0xFFFFFFFF:08X}"


def select_logged_channel(
    stereo: np.ndarray,
    *,
    channel_mode: str = CHANNEL_MODE_AUTO,
    sample_shift_bits: int = 0,
) -> tuple[np.ndarray, str]:
    mono, used_mode = select_mono_channel(stereo, channel_mode=channel_mode)
    if sample_shift_bits:
        mono = np.right_shift(mono, sample_shift_bits)
    return mono.astype(np.int32, copy=False), used_mode


def write_csv_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "timestamp_ns",
            "sequence",
            "left_i32",
            "right_i32",
            "left_hex",
            "right_hex",
            "mono_i32",
            "channel_used",
            "abs_left",
            "abs_right",
            "abs_mono",
        ]
    )


def write_csv_rows(
    writer: csv.writer,
    stereo: np.ndarray,
    mono: np.ndarray,
    channel_used: str,
    next_sequence: int,
    *,
    timestamp_ns_fn: Callable[[], int] = time.time_ns,
) -> int:
    stereo_i32 = np.asarray(stereo, dtype=np.int32)
    mono_i32 = np.asarray(mono, dtype=np.int32).reshape(-1)
    if stereo_i32.ndim != 2 or stereo_i32.shape[1] != 2:
        stereo_i32 = stereo_i32.reshape(-1, 2)
    if mono_i32.shape[0] != stereo_i32.shape[0]:
        raise ValueError("mono sample count must match stereo frame count")

    for idx, pair in enumerate(stereo_i32):
        left = int(pair[0])
        right = int(pair[1])
        mono_value = int(mono_i32[idx])
        writer.writerow(
            [
                int(timestamp_ns_fn()),
                int(next_sequence),
                left,
                right,
                format_i32_hex(left),
                format_i32_hex(right),
                mono_value,
                str(channel_used),
                abs(left),
                abs(right),
                abs(mono_value),
            ]
        )
        next_sequence += 1

    return next_sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the mirrored microphone I2S stream from ALSA and log raw stereo samples."
    )
    parser.add_argument("-D", "--device", default=DEFAULT_AUDIO_DEVICE, help="ALSA device, ex.: hw:2,0")
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "arecord", "alsa-c"),
        default=DEFAULT_CAPTURE_BACKEND,
        help="Backend used to capture the ALSA stream",
    )
    parser.add_argument(
        "--capture-binary",
        default=None,
        help="Path to the compiled native C capture helper",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=DEFAULT_CAPTURE_RATE_HZ,
        help="Host-side ALSA sample rate in Hz",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_LOGGER_CHUNK_FRAMES,
        help="Stereo frames read per chunk",
    )
    parser.add_argument(
        "--channel-mode",
        choices=(CHANNEL_MODE_AUTO, CHANNEL_MODE_LEFT, CHANNEL_MODE_RIGHT, CHANNEL_MODE_AVERAGE),
        default=CHANNEL_MODE_AUTO,
        help="Channel used to derive the logged mono column",
    )
    parser.add_argument(
        "--sample-shift-bits",
        type=int,
        default=0,
        help="Arithmetic right shift applied to the derived mono column",
    )
    parser.add_argument("--csv", type=Path, default=None, help="CSV destination path")
    parser.add_argument("--raw-out", type=Path, default=None, help="Raw S32_LE stereo dump path")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Capture duration in seconds; 0 runs until Ctrl+C",
    )
    parser.add_argument(
        "--csv-flush-every-chunks",
        type=int,
        default=DEFAULT_CSV_FLUSH_EVERY_CHUNKS,
        help="Flush the CSV file every N chunks",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")
    if args.sample_shift_bits < 0:
        parser.error("--sample-shift-bits must be non-negative")
    if args.seconds < 0:
        parser.error("--seconds must be non-negative")
    if args.csv is None and args.raw_out is None:
        parser.error("at least one of --csv or --raw-out must be provided")
    if args.csv_flush_every_chunks <= 0:
        parser.error("--csv-flush-every-chunks must be positive")

    try:
        device = resolve_audio_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    csv_path = args.csv.resolve() if args.csv is not None else None
    raw_path = args.raw_out.resolve() if args.raw_out is not None else None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_capture_cmd(
        device,
        args.rate,
        backend=args.capture_backend,
        capture_binary=args.capture_binary,
    )
    proc = start_capture_process(
        device,
        args.rate,
        backend=args.capture_backend,
        capture_binary=args.capture_binary,
    )
    print("Starting:", " ".join(cmd), flush=True)

    stop_requested = False

    def _request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _request_stop)

    chunk_bytes = args.chunk_frames * 8
    chunk_index = 0
    next_sequence = 0
    total_frames = 0
    channel_counts = {
        CHANNEL_MODE_LEFT: 0,
        CHANNEL_MODE_RIGHT: 0,
        CHANNEL_MODE_AVERAGE: 0,
    }
    start_time = time.monotonic()
    csv_handle = None
    raw_handle = None

    try:
        if csv_path is not None:
            csv_handle = csv_path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(csv_handle)
            write_csv_header(writer)
        else:
            writer = None

        if raw_path is not None:
            raw_handle = raw_path.open("wb")

        while not stop_requested:
            if args.seconds > 0.0 and (time.monotonic() - start_time) >= args.seconds:
                break

            if proc.stdout is None:
                raise RuntimeError("capture process stdout is not available")
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                if proc.poll() is not None:
                    break
                continue

            trimmed = trim_incomplete_frames(chunk, bytes_per_frame=8)
            if not trimmed:
                continue

            stereo = decode_stereo_frames(trimmed)
            if stereo.size == 0:
                continue

            if raw_handle is not None:
                raw_handle.write(trimmed)

            mono, channel_used = select_logged_channel(
                stereo,
                channel_mode=args.channel_mode,
                sample_shift_bits=args.sample_shift_bits,
            )
            channel_counts[channel_used] = int(channel_counts.get(channel_used, 0)) + int(stereo.shape[0])

            if writer is not None:
                next_sequence = write_csv_rows(
                    writer,
                    stereo,
                    mono,
                    channel_used,
                    next_sequence,
                )
                chunk_index += 1
                if chunk_index % args.csv_flush_every_chunks == 0:
                    csv_handle.flush()

            total_frames += int(stereo.shape[0])
    finally:
        stop_process(proc)
        if csv_handle is not None:
            csv_handle.flush()
            csv_handle.close()
        if raw_handle is not None:
            raw_handle.flush()
            raw_handle.close()

    elapsed = max(1e-9, time.monotonic() - start_time)
    print(
        "Capture completed:",
        f"frames={total_frames}",
        f"seconds={elapsed:.3f}",
        f"effective_rate={total_frames / elapsed:.1f} fps",
        f"channel_counts={channel_counts}",
        flush=True,
    )
    if csv_path is not None:
        print("CSV:", csv_path, flush=True)
    if raw_path is not None:
        print("Raw:", raw_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
