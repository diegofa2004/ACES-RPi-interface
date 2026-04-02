import argparse
import csv
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .i2s_stream import DEFAULT_CAPTURE_RATE_HZ
except ImportError:
    from i2s_stream import DEFAULT_CAPTURE_RATE_HZ


SCRIPT_DIR = Path(__file__).resolve().parent
_FALLBACK_PALETTE_ANCHORS = np.array(
    [
        [0, 0, 4],
        [31, 12, 72],
        [85, 15, 109],
        [136, 34, 106],
        [186, 54, 85],
        [227, 89, 51],
        [249, 140, 10],
        [252, 195, 65],
        [252, 255, 164],
    ],
    dtype=np.float32,
)


def _display_available() -> bool:
    return any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET"))


def _load_pyplot(requested_backend: str):
    try:
        import matplotlib
    except ImportError as exc:
        backend_name = (requested_backend or "auto").strip()
        if backend_name.lower() in ("auto", "agg"):
            return None, "builtin-png", False
        raise SystemExit(
            "matplotlib is required for interactive plotFFT.py usage. "
            "Use --backend Agg or install the project requirements first."
        ) from exc

    backend_name = (requested_backend or "auto").strip()
    if backend_name.lower() == "auto":
        candidates = ["TkAgg", "Agg"] if _display_available() else ["Agg"]
    elif backend_name.lower() == "agg":
        candidates = ["Agg"]
    else:
        candidates = [backend_name, "Agg"]

    last_error = None
    for backend in candidates:
        try:
            matplotlib.use(backend, force=True)
            sys.modules.pop("matplotlib.pyplot", None)
            import matplotlib.pyplot as plt

            return plt, backend, backend.lower() != "agg"
        except Exception as exc:  # pragma: no cover - backend availability depends on target system
            last_error = exc

    raise SystemExit(f"Unable to initialize matplotlib backend {backend_name!r}: {last_error}")


def _prepare_fft_history(fft_cache: np.ndarray) -> Optional[np.ndarray]:
    fft_array = np.asarray(fft_cache, dtype=np.float32)
    if fft_array.ndim == 1:
        fft_array = fft_array.reshape(1, -1)
    if fft_array.ndim != 2 or fft_array.shape[0] == 0 or fft_array.shape[1] == 0:
        return None
    return fft_array


def _frequency_axis(rate: int, frame_bins: int, max_bin: int) -> tuple[np.ndarray, float]:
    freq_per_bin = rate / frame_bins
    freqs_hz = np.arange(max_bin, dtype=np.float32) * freq_per_bin
    return freqs_hz, freq_per_bin


def _compute_fft_db(fft_cache: np.ndarray, rate: int, frame_bins: int, max_freq: float) -> tuple[np.ndarray, np.ndarray, float]:
    freq_per_bin = rate / frame_bins
    max_bin = min(fft_cache.shape[1], int(max_freq / freq_per_bin) + 1)
    fft_mag = np.abs(fft_cache[:, :max_bin])
    fft_mag = 20.0 * np.log10(fft_mag + 1e-9)
    freqs_hz, _ = _frequency_axis(rate, frame_bins, max_bin)
    return fft_mag, freqs_hz, freq_per_bin


def _select_representative_frame(fft_cache: np.ndarray, sample_mode: str) -> tuple[np.ndarray, int]:
    if sample_mode == "latest":
        return fft_cache[-1], fft_cache.shape[0] - 1
    if sample_mode == "mean":
        return np.mean(fft_cache, axis=0), -1

    frame_energy = np.sum(np.square(fft_cache), axis=1)
    frame_idx = int(np.argmax(frame_energy))
    return fft_cache[frame_idx], frame_idx


def _select_window_frame(
    fft_cache: np.ndarray,
    sample_mode: str,
    window_index: Optional[int],
) -> tuple[np.ndarray, int]:
    if window_index is None:
        return _select_representative_frame(fft_cache, sample_mode)

    frame_count = fft_cache.shape[0]
    resolved_index = window_index if window_index >= 0 else frame_count + window_index
    if resolved_index < 0 or resolved_index >= frame_count:
        raise IndexError(f"window index {window_index} is out of range for {frame_count} FFT frames")
    return fft_cache[resolved_index], resolved_index


def _centered_fft_view(fft_frame: np.ndarray, rate: int, frame_bins: int) -> tuple[np.ndarray, np.ndarray]:
    frame = np.asarray(fft_frame, dtype=np.float32).reshape(-1)
    if frame.size == 0:
        raise ValueError("Cannot center an empty FFT frame")

    if frame.size >= frame_bins:
        truncated = frame[:frame_bins]
        centered_freqs = np.fft.fftshift(np.fft.fftfreq(frame_bins, d=1.0 / float(rate))).astype(np.float32)
        centered_values = np.fft.fftshift(truncated).astype(np.float32)
        return centered_freqs, centered_values

    freq_per_bin = float(rate) / float(frame_bins)
    negative = frame[1:][::-1]
    centered_values = np.concatenate([negative, frame]).astype(np.float32, copy=False)
    negative_freqs = -np.arange(frame.size - 1, 0, -1, dtype=np.float32) * freq_per_bin
    positive_freqs = np.arange(frame.size, dtype=np.float32) * freq_per_bin
    centered_freqs = np.concatenate([negative_freqs, positive_freqs]).astype(np.float32, copy=False)
    return centered_freqs, centered_values


def _scale_fft_values(fft_values: np.ndarray, amplitude_scale: str) -> tuple[np.ndarray, str]:
    values = np.asarray(fft_values, dtype=np.float32)
    if amplitude_scale == "db":
        return 20.0 * np.log10(np.abs(values) + 1e-9), "Magnitude (dB)"
    return values, "Magnitude"


def _load_mock_fft_frame(mock_fft_csv: Path, example_idx: int) -> np.ndarray:
    bins: dict[int, float] = {}
    with mock_fft_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_example = int(row["example_idx"])
            if row_example != example_idx:
                continue
            bin_idx = int(row["bin_idx"])
            expected_real = float(row["expected_real"])
            expected_imag = float(row["expected_imag"])
            bins[bin_idx] = float(np.hypot(expected_real, expected_imag))

    if not bins:
        raise ValueError(f"No FFT rows found for example {example_idx} in {mock_fft_csv}")

    frame = np.zeros(max(bins) + 1, dtype=np.float32)
    for bin_idx, magnitude in bins.items():
        frame[bin_idx] = magnitude
    return frame


def _render_single_window_plot(
    spectrum_ax,
    fft_frame: np.ndarray,
    rate: int,
    frame_bins: int,
    max_freq: float,
    *,
    amplitude_scale: str,
    center_zero: bool,
    sample_mode: str,
    frame_index: int,
    mock_fft_frame: Optional[np.ndarray] = None,
) -> None:
    spectrum_ax.clear()

    if center_zero:
        plot_freqs_hz, plot_values = _centered_fft_view(fft_frame, rate, frame_bins)
        x_limit = min(max_freq, float(np.max(np.abs(plot_freqs_hz)))) if plot_freqs_hz.size else max_freq
    else:
        plot_freqs_hz, _ = _frequency_axis(rate, frame_bins, fft_frame.size)
        plot_values = np.asarray(fft_frame, dtype=np.float32)
        x_limit = min(max_freq, float(plot_freqs_hz[-1])) if plot_freqs_hz.size else max_freq

    plot_values, ylabel = _scale_fft_values(plot_values, amplitude_scale)
    spectrum_ax.plot(plot_freqs_hz, plot_values, color="#cc5500", linewidth=1.8, label="capturado")

    if mock_fft_frame is not None:
        expected_frame = np.asarray(mock_fft_frame[: fft_frame.size], dtype=np.float32)
        if center_zero:
            mock_freqs_hz, mock_values = _centered_fft_view(expected_frame, rate, frame_bins)
        else:
            mock_freqs_hz, _ = _frequency_axis(rate, frame_bins, expected_frame.size)
            mock_values = expected_frame
        mock_values, _ = _scale_fft_values(mock_values, amplitude_scale)
        spectrum_ax.plot(
            mock_freqs_hz,
            mock_values,
            color="#004c99",
            linewidth=1.4,
            linestyle="--",
            label="mock esperado",
        )
        spectrum_ax.legend(loc="best")

    if center_zero:
        spectrum_ax.set_xlim(-x_limit, x_limit)
        spectrum_ax.set_xlabel("Frequencia (Hz, centrada em 0)")
    else:
        spectrum_ax.set_xlim(0.0, x_limit)
        spectrum_ax.set_xlabel("Frequencia (Hz)")
    spectrum_ax.set_ylabel(ylabel)
    spectrum_ax.grid(True, which="both", alpha=0.25)

    if frame_index >= 0:
        title_suffix = f"frame {frame_index}"
    else:
        title_suffix = sample_mode
    center_label = "centrada em 0" if center_zero else "faixa positiva"
    spectrum_ax.set_title(f"Janela unica da FFT ({center_label}, {title_suffix})")


def _render_plot(
    spectrum_ax,
    spectrogram_ax,
    fft_cache: np.ndarray,
    rate: int,
    frame_bins: int,
    max_freq: float,
    step_hz: float,
    *,
    sample_mode: str,
) -> None:
    fft_mag, freqs_hz, freq_per_bin = _compute_fft_db(fft_cache, rate, frame_bins, max_freq)
    max_bin = fft_mag.shape[1]
    representative_frame, representative_idx = _select_representative_frame(fft_cache[:, :max_bin], sample_mode)
    representative_db = 20.0 * np.log10(np.abs(representative_frame) + 1e-9)

    vmax = float(np.max(fft_mag))
    vmin = vmax - 50.0

    spectrum_ax.clear()
    spectrum_ax.plot(freqs_hz, representative_db, color="#cc5500", linewidth=1.8)
    spectrum_ax.set_xlim(0.0, freqs_hz[-1] if freqs_hz.size else max_freq)
    spectrum_ax.set_ylabel("Magnitude (dB)")
    spectrum_ax.grid(True, which="both", alpha=0.25)
    if representative_idx >= 0:
        spectrum_ax.set_title(f"Espectro FFT representativo ({sample_mode}, frame {representative_idx})")
    else:
        spectrum_ax.set_title(f"Espectro FFT representativo ({sample_mode})")

    spectrogram_ax.clear()
    spectrogram_ax.imshow(
        fft_mag.T,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    spectrogram_ax.set_title("FFT ao longo do tempo")
    spectrogram_ax.set_xlabel("Tempo (frames)")
    spectrogram_ax.set_ylabel("Frequencia (Hz)")

    freq_limit = min(max_freq, (max_bin - 1) * freq_per_bin)
    tick_freqs_hz = np.arange(0.0, freq_limit + step_hz, step_hz)
    yticks = tick_freqs_hz / freq_per_bin
    valid = yticks < max_bin
    spectrogram_ax.set_yticks(yticks[valid])
    spectrogram_ax.set_yticklabels([f"{int(freq)}" for freq in tick_freqs_hz[valid]])


def _build_fallback_palette() -> np.ndarray:
    color_positions = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    anchor_positions = np.linspace(0.0, 1.0, len(_FALLBACK_PALETTE_ANCHORS), dtype=np.float32)
    channels = [
        np.interp(color_positions, anchor_positions, _FALLBACK_PALETTE_ANCHORS[:, idx])
        for idx in range(3)
    ]
    return np.stack(channels, axis=1).astype(np.uint8)


def _write_png_rgb(output_path: Path, rgb: np.ndarray) -> None:
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("RGB PNG writer expects an array shaped as (height, width, 3)")

    height, width, _ = rgb_u8.shape
    scanlines = b"".join(b"\x00" + rgb_u8[row].tobytes() for row in range(height))
    compressed = zlib.compress(scanlines, level=9)

    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    png_bytes = b"\x89PNG\r\n\x1a\n"
    png_bytes += _chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
    )
    png_bytes += _chunk(b"IDAT", compressed)
    png_bytes += _chunk(b"IEND", b"")
    output_path.write_bytes(png_bytes)


def _render_png_fallback(
    fft_cache: np.ndarray,
    rate: int,
    frame_bins: int,
    max_freq: float,
    output_path: Path,
) -> None:
    fft_mag, _, _ = _compute_fft_db(fft_cache, rate, frame_bins, max_freq)

    vmax = float(np.max(fft_mag))
    vmin = vmax - 50.0
    if vmax <= vmin:
        normalized = np.zeros_like(fft_mag, dtype=np.float32)
    else:
        normalized = np.clip((fft_mag - vmin) / (vmax - vmin), 0.0, 1.0)

    palette = _build_fallback_palette()
    color_index = np.rint(normalized * 255.0).astype(np.uint8)
    rgb = palette[color_index].transpose(1, 0, 2)
    rgb = rgb[::-1, :, :]

    scale_x = 4 if rgb.shape[1] < 256 else (2 if rgb.shape[1] < 512 else 1)
    scale_y = 2 if rgb.shape[0] < 256 else 1
    if scale_y > 1:
        rgb = np.repeat(rgb, scale_y, axis=0)
    if scale_x > 1:
        rgb = np.repeat(rgb, scale_x, axis=1)

    _write_png_rgb(output_path, rgb)


def _load_fft_array(fft_path: Path) -> Optional[np.ndarray]:
    try:
        raw_fft_cache = np.load(fft_path)
    except (OSError, ValueError, EOFError) as exc:
        print(f"Falha ao carregar {fft_path}: {exc}", flush=True)
        return None

    fft_cache = _prepare_fft_history(raw_fft_cache)
    if fft_cache is None:
        print("fft.npy tem formato invalido para plot:", np.asarray(raw_fft_cache).shape, flush=True)
        return None

    return fft_cache


def _resolve_cli_defaults(args: argparse.Namespace) -> dict[str, object]:
    preset = getattr(args, "mode_preset", None)

    if args.plot_mode is not None:
        plot_mode = args.plot_mode
    elif preset == "capture-window":
        plot_mode = "window"
    else:
        plot_mode = "history"

    if args.sample_mode is not None:
        sample_mode = args.sample_mode
    elif plot_mode == "window":
        sample_mode = "latest"
    else:
        sample_mode = "max-energy"

    if args.output_file is not None:
        output_file = args.output_file
    elif plot_mode == "window":
        output_file = SCRIPT_DIR / "fft_single_window.png"
    else:
        output_file = SCRIPT_DIR / "fft_latest.png"

    if args.capture_next_window is not None:
        capture_next_window = args.capture_next_window
    else:
        capture_next_window = plot_mode == "window"

    if args.center_zero is not None:
        center_zero = args.center_zero
    else:
        center_zero = plot_mode == "window"

    return {
        "plot_mode": plot_mode,
        "sample_mode": sample_mode,
        "output_file": output_file,
        "capture_next_window": capture_next_window,
        "center_zero": center_zero,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch fft.npy generated by analyzer_from_fpga_fft.py and plot either the live spectrogram view or a single FFT window."
    )
    parser.add_argument(
        "--fft-file",
        type=Path,
        default=SCRIPT_DIR / "fft.npy",
        help="Path to fft.npy generated by the analyzer",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_CAPTURE_RATE_HZ,
        help="Host-side ALSA sample rate in Hz (nominal wire rate is 48828.125 Hz)",
    )
    parser.add_argument("--frame-bins", type=int, default=512, help="Complex bins per FPGA FFT frame")
    parser.add_argument("--max-freq", type=float, default=20000.0, help="Maximum frequency shown on the plot")
    parser.add_argument("--step-hz", type=float, default=1000.0, help="Y-axis label spacing in Hz")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--plot-mode",
        choices=("history", "window"),
        default=None,
        help="Advanced selector for plot mode: full FFT history or a single FFT window",
    )
    mode_group.add_argument(
        "--spectrogram",
        dest="mode_preset",
        action="store_const",
        const="spectrogram",
        help="Convenience preset for the live spectrum + spectrogram view",
    )
    mode_group.add_argument(
        "--capture-window",
        dest="mode_preset",
        action="store_const",
        const="capture-window",
        help="Convenience preset for capturing the next FFT window centered at 0 Hz",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Matplotlib backend. Use 'auto' for GUI when available and 'Agg' when headless.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="PNG output path (default: fft_latest.png for spectrogram, fft_single_window.png for single-window capture)",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("latest", "max-energy", "mean"),
        default=None,
        help="Representative FFT frame selection (default: max-energy for spectrogram, latest for window capture)",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=None,
        help="Explicit FFT frame index used in --plot-mode window (negative values count from the end)",
    )
    parser.add_argument(
        "--capture-next-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In window mode, wait for the next fft.npy update before plotting once and exiting",
    )
    parser.add_argument(
        "--center-zero",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In window mode, plot the FFT centered at 0 Hz with negative and positive frequencies",
    )
    parser.add_argument(
        "--window-scale",
        choices=("linear", "db"),
        default="linear",
        help="Amplitude scale used by --plot-mode window",
    )
    parser.add_argument(
        "--mock-fft-csv",
        type=Path,
        default=None,
        help="Optional expected FFT CSV to overlay on the single-window plot",
    )
    parser.add_argument(
        "--mock-example",
        type=int,
        default=0,
        help="Example index used with --mock-fft-csv",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.10, help="Polling interval for fft.npy updates")
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.frame_bins <= 0:
        parser.error("--frame-bins must be positive")
    if args.max_freq <= 0:
        parser.error("--max-freq must be positive")
    if args.step_hz <= 0:
        parser.error("--step-hz must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    cli_cfg = _resolve_cli_defaults(args)
    plot_mode = str(cli_cfg["plot_mode"])
    sample_mode = str(cli_cfg["sample_mode"])
    output_file = Path(cli_cfg["output_file"])
    capture_next_window = bool(cli_cfg["capture_next_window"])
    center_zero = bool(cli_cfg["center_zero"])

    if capture_next_window and plot_mode != "window":
        parser.error("--capture-next-window requires --plot-mode window")
    if args.window_index is not None and plot_mode != "window":
        parser.error("--window-index requires --plot-mode window")
    if center_zero and plot_mode != "window":
        parser.error("--center-zero requires --plot-mode window")
    if args.mock_fft_csv is not None and plot_mode != "window":
        parser.error("--mock-fft-csv requires --plot-mode window")

    plt, backend_name, interactive = _load_pyplot(args.backend)

    fft_path = args.fft_file.resolve()
    output_path = output_file.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mock_fft_frame = None
    if args.mock_fft_csv is not None:
        try:
            mock_fft_frame = _load_mock_fft_frame(args.mock_fft_csv.resolve(), args.mock_example)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Unable to load --mock-fft-csv: {exc}") from exc

    if plt is not None:
        if plot_mode == "window":
            fig, spectrum_ax = plt.subplots(1, 1, figsize=(10, 4.8))
            spectrogram_ax = None
        else:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(10, 8),
                gridspec_kw={"height_ratios": [1.1, 1.8]},
            )
            spectrum_ax, spectrogram_ax = axes
        if interactive:
            plt.ion()
            plt.show(block=False)
            print(f"Using matplotlib backend: {backend_name}", flush=True)
        else:
            print(
                f"Using matplotlib backend: {backend_name} (headless mode, updating {output_path})",
                flush=True,
            )
    else:
        if plot_mode == "window":
            raise SystemExit("Single-window FFT plots require matplotlib. Install it or choose another backend.")
        fig = None
        spectrum_ax = None
        spectrogram_ax = None
        print(
            f"Using builtin headless PNG renderer (updating {output_path})",
            flush=True,
        )

    fft_cache = None
    if capture_next_window and fft_path.exists():
        last_mtime = fft_path.stat().st_mtime
        print(f"Waiting for the next FFT update after {fft_path}...", flush=True)
    else:
        last_mtime = 0.0

    while True:
        if not fft_path.exists():
            if interactive:
                plt.pause(args.poll_seconds)
            else:
                time.sleep(args.poll_seconds)
            continue

        mtime = fft_path.stat().st_mtime
        if fft_cache is not None and mtime == last_mtime:
            if interactive:
                plt.pause(args.poll_seconds)
            else:
                time.sleep(args.poll_seconds)
            continue

        if capture_next_window and mtime == last_mtime:
            if interactive:
                plt.pause(args.poll_seconds)
            else:
                time.sleep(args.poll_seconds)
            continue

        print("FFT atualizada:", fft_path, flush=True)
        fft_cache = _load_fft_array(fft_path)
        last_mtime = mtime

        if fft_cache is None:
            if interactive:
                plt.pause(args.poll_seconds)
            else:
                time.sleep(args.poll_seconds)
            continue

        if plot_mode == "window":
            assert plt is not None
            assert fig is not None
            assert spectrum_ax is not None

            try:
                fft_frame, frame_index = _select_window_frame(fft_cache, sample_mode, args.window_index)
            except IndexError as exc:
                raise SystemExit(str(exc)) from exc

            peak_bin = int(np.argmax(fft_frame))
            peak_freq_hz = float(peak_bin) * float(args.rate) / float(args.frame_bins)
            print(
                f"Janela FFT capturada: frame={frame_index} peak_bin={peak_bin} peak_freq_hz={peak_freq_hz:.3f}",
                flush=True,
            )

            _render_single_window_plot(
                spectrum_ax,
                fft_frame,
                args.rate,
                args.frame_bins,
                args.max_freq,
                amplitude_scale=args.window_scale,
                center_zero=center_zero,
                sample_mode=sample_mode,
                frame_index=frame_index,
                mock_fft_frame=mock_fft_frame,
            )
            fig.tight_layout()

            if interactive:
                plt.ioff()
                plt.show()
            else:
                fig.savefig(output_path, dpi=120)
                print("PNG atualizada:", output_path, flush=True)
            return 0

        if plt is not None and spectrum_ax is not None and spectrogram_ax is not None and fig is not None:
            _render_plot(
                spectrum_ax,
                spectrogram_ax,
                fft_cache,
                args.rate,
                args.frame_bins,
                args.max_freq,
                args.step_hz,
                sample_mode=sample_mode,
            )
            fig.tight_layout()

            if interactive:
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
            else:
                fig.savefig(output_path, dpi=120)
                print("PNG atualizada:", output_path, flush=True)
        else:
            _render_png_fallback(
                fft_cache,
                rate=args.rate,
                frame_bins=args.frame_bins,
                max_freq=args.max_freq,
                output_path=output_path,
            )
            print("PNG atualizada:", output_path, flush=True)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
