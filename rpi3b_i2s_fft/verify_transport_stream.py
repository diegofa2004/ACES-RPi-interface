#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from .fpga_fft_adapter import (
        CHANNEL_MODE_AUTO,
        CHANNEL_MODE_AVERAGE,
        CHANNEL_MODE_LEFT,
        CHANNEL_MODE_RIGHT,
        compute_fft_magnitude,
        decode_stereo_frames,
        prepare_audio_window,
    )
except ImportError:
    from fpga_fft_adapter import (
        CHANNEL_MODE_AUTO,
        CHANNEL_MODE_AVERAGE,
        CHANNEL_MODE_LEFT,
        CHANNEL_MODE_RIGHT,
        compute_fft_magnitude,
        decode_stereo_frames,
        prepare_audio_window,
    )


@dataclass(frozen=True)
class RawTransportConfig:
    sample_rate: int = 48828
    frame_bins: int = 512
    useful_bins: int = 256
    channel_mode: str = CHANNEL_MODE_AUTO
    sample_shift_bits: int = 0
    remove_dc: bool = True


@dataclass(frozen=True)
class WindowReport:
    window_index: int
    channel_used: str
    rms: float
    mean_abs: float
    peak_bin: int
    peak_frequency_hz: float
    peak_magnitude: float
    left_mean_abs: float
    right_mean_abs: float
    correlation_lr: float


@dataclass(frozen=True)
class TransportSummary:
    sample_count: int
    window_count: int
    left_mean_abs: float
    right_mean_abs: float
    left_nonzero_ratio: float
    right_nonzero_ratio: float
    mean_correlation_lr: float
    dominant_channel: str
    channel_counts: dict[str, int]
    peak_frequency_hz: float
    frame_reports: tuple[WindowReport, ...]


def _safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    left_f64 = np.asarray(left, dtype=np.float64)
    right_f64 = np.asarray(right, dtype=np.float64)
    if left_f64.size == 0 or right_f64.size == 0:
        return 0.0
    if np.std(left_f64) == 0.0 or np.std(right_f64) == 0.0:
        return 0.0
    return float(np.corrcoef(left_f64, right_f64)[0, 1])


def load_raw_capture(raw_path: Path) -> np.ndarray:
    return decode_stereo_frames(raw_path.read_bytes())


def analyze_raw_stereo_frames(
    stereo: np.ndarray,
    cfg: RawTransportConfig,
    *,
    print_limit: int = 8,
    printer: Callable[[str], None] = print,
) -> TransportSummary:
    stereo_i32 = np.asarray(stereo, dtype=np.int32)
    if stereo_i32.size == 0:
        return TransportSummary(
            sample_count=0,
            window_count=0,
            left_mean_abs=0.0,
            right_mean_abs=0.0,
            left_nonzero_ratio=0.0,
            right_nonzero_ratio=0.0,
            mean_correlation_lr=0.0,
            dominant_channel=CHANNEL_MODE_LEFT,
            channel_counts={CHANNEL_MODE_LEFT: 0, CHANNEL_MODE_RIGHT: 0, CHANNEL_MODE_AVERAGE: 0},
            peak_frequency_hz=0.0,
            frame_reports=tuple(),
        )

    if stereo_i32.ndim != 2 or stereo_i32.shape[1] != 2:
        stereo_i32 = stereo_i32.reshape(-1, 2)

    left = stereo_i32[:, 0]
    right = stereo_i32[:, 1]
    sample_count = int(stereo_i32.shape[0])
    left_mean_abs = float(np.mean(np.abs(left.astype(np.int64)), dtype=np.float64))
    right_mean_abs = float(np.mean(np.abs(right.astype(np.int64)), dtype=np.float64))
    left_nonzero_ratio = float(np.count_nonzero(left) / max(1, sample_count))
    right_nonzero_ratio = float(np.count_nonzero(right) / max(1, sample_count))

    usable_samples = (sample_count // cfg.frame_bins) * cfg.frame_bins
    window_count = usable_samples // cfg.frame_bins
    frame_reports: list[WindowReport] = []
    correlation_values: list[float] = []
    channel_counts = {
        CHANNEL_MODE_LEFT: 0,
        CHANNEL_MODE_RIGHT: 0,
        CHANNEL_MODE_AVERAGE: 0,
    }

    for window_index in range(window_count):
        start = window_index * cfg.frame_bins
        stop = start + cfg.frame_bins
        window_stereo = stereo_i32[start:stop]
        prepared_window, channel_used = prepare_audio_window(
            window_stereo,
            channel_mode=cfg.channel_mode,
            sample_shift_bits=cfg.sample_shift_bits,
            remove_dc=cfg.remove_dc,
        )
        channel_counts[channel_used] = int(channel_counts.get(channel_used, 0)) + 1
        fft_full, fft_useful = compute_fft_magnitude(
            prepared_window,
            frame_bins=cfg.frame_bins,
            useful_bins=cfg.useful_bins,
        )
        if fft_useful.size > 1:
            peak_bin = int(np.argmax(fft_useful[1:]) + 1)
        elif fft_useful.size == 1:
            peak_bin = 0
        else:
            peak_bin = -1
        peak_magnitude = float(fft_useful[peak_bin]) if peak_bin >= 0 and fft_useful.size else 0.0
        peak_frequency_hz = float(peak_bin * cfg.sample_rate / cfg.frame_bins) if peak_bin >= 0 else 0.0
        left_chunk = window_stereo[:, 0]
        right_chunk = window_stereo[:, 1]
        correlation_lr = _safe_corrcoef(left_chunk, right_chunk)
        correlation_values.append(correlation_lr)
        rms = float(np.sqrt(np.mean(np.square(prepared_window), dtype=np.float32)))
        mean_abs = float(np.mean(np.abs(prepared_window), dtype=np.float32))

        report = WindowReport(
            window_index=window_index,
            channel_used=channel_used,
            rms=rms,
            mean_abs=mean_abs,
            peak_bin=peak_bin,
            peak_frequency_hz=peak_frequency_hz,
            peak_magnitude=peak_magnitude,
            left_mean_abs=float(np.mean(np.abs(left_chunk.astype(np.int64)), dtype=np.float64)),
            right_mean_abs=float(np.mean(np.abs(right_chunk.astype(np.int64)), dtype=np.float64)),
            correlation_lr=correlation_lr,
        )
        frame_reports.append(report)

        if window_index < print_limit:
            printer(
                "window="
                f"{window_index} channel={channel_used} "
                f"rms={rms:.3f} peak_bin={peak_bin} peak_hz={peak_frequency_hz:.1f} "
                f"corr_lr={correlation_lr:.3f}"
            )

        del fft_full

    dominant_channel = max(channel_counts, key=channel_counts.get) if channel_counts else CHANNEL_MODE_LEFT
    peak_frequency_hz = (
        float(np.median(np.asarray([report.peak_frequency_hz for report in frame_reports], dtype=np.float64)))
        if frame_reports
        else 0.0
    )

    return TransportSummary(
        sample_count=sample_count,
        window_count=window_count,
        left_mean_abs=left_mean_abs,
        right_mean_abs=right_mean_abs,
        left_nonzero_ratio=left_nonzero_ratio,
        right_nonzero_ratio=right_nonzero_ratio,
        mean_correlation_lr=float(np.mean(np.asarray(correlation_values, dtype=np.float64))) if correlation_values else 0.0,
        dominant_channel=dominant_channel,
        channel_counts=channel_counts,
        peak_frequency_hz=peak_frequency_hz,
        frame_reports=tuple(frame_reports),
    )


def summary_to_dict(summary: TransportSummary) -> dict[str, object]:
    payload = asdict(summary)
    payload["frame_reports"] = [asdict(report) for report in summary.frame_reports]
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the raw mirrored microphone transport captured from the FPGA."
    )
    parser.add_argument("--raw", type=Path, required=True, help="Raw S32_LE stereo capture file")
    parser.add_argument("-r", "--rate", type=int, default=48828, help="Sample rate in Hz")
    parser.add_argument("--frame-bins", type=int, default=512, help="Samples per FFT window")
    parser.add_argument("--useful-bins", type=int, default=256, help="Magnitude bins kept per window")
    parser.add_argument(
        "--channel-mode",
        choices=(CHANNEL_MODE_AUTO, CHANNEL_MODE_LEFT, CHANNEL_MODE_RIGHT, CHANNEL_MODE_AVERAGE),
        default=CHANNEL_MODE_AUTO,
        help="Channel used for the verification FFT windows",
    )
    parser.add_argument("--sample-shift-bits", type=int, default=0, help="Arithmetic right shift applied before FFT")
    parser.add_argument("--keep-dc", action="store_true", help="Keep the DC component before FFT")
    parser.add_argument("--print-limit", type=int, default=8, help="Number of windows printed to stdout")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON summary destination")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-window summary")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.frame_bins <= 1:
        parser.error("--frame-bins must be greater than 1")
    if args.useful_bins <= 0:
        parser.error("--useful-bins must be positive")
    if args.sample_shift_bits < 0:
        parser.error("--sample-shift-bits must be non-negative")

    cfg = RawTransportConfig(
        sample_rate=args.rate,
        frame_bins=args.frame_bins,
        useful_bins=args.useful_bins,
        channel_mode=args.channel_mode,
        sample_shift_bits=args.sample_shift_bits,
        remove_dc=not args.keep_dc,
    )

    stereo = load_raw_capture(args.raw.resolve())
    summary = analyze_raw_stereo_frames(
        stereo,
        cfg,
        print_limit=max(0, int(args.print_limit)),
        printer=(lambda _line: None) if args.quiet else print,
    )

    if not args.quiet:
        print(
            "summary:",
            f"samples={summary.sample_count}",
            f"windows={summary.window_count}",
            f"dominant_channel={summary.dominant_channel}",
            f"left_mean_abs={summary.left_mean_abs:.3f}",
            f"right_mean_abs={summary.right_mean_abs:.3f}",
            f"left_nonzero_ratio={summary.left_nonzero_ratio:.3f}",
            f"right_nonzero_ratio={summary.right_nonzero_ratio:.3f}",
            f"corr_lr={summary.mean_correlation_lr:.3f}",
            f"peak_hz={summary.peak_frequency_hz:.1f}",
        )

    if args.json_out is not None:
        json_path = args.json_out.resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary_to_dict(summary), indent=2, sort_keys=True), encoding="utf-8")

    return 0 if summary.sample_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
