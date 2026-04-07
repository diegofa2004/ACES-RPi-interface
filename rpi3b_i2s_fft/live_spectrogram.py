import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .fpga_fft_adapter import (
        CHANNEL_MODE_AUTO,
        CHANNEL_MODE_AVERAGE,
        CHANNEL_MODE_LEFT,
        CHANNEL_MODE_RIGHT,
        FFTAdapterConfig,
        FPGAFFTReceiver,
    )
    from .i2s_stream import AUTO_AUDIO_DEVICE, DEFAULT_CAPTURE_BACKEND, DEFAULT_CAPTURE_RATE_HZ, resolve_audio_device
except ImportError:
    from fpga_fft_adapter import (
        CHANNEL_MODE_AUTO,
        CHANNEL_MODE_AVERAGE,
        CHANNEL_MODE_LEFT,
        CHANNEL_MODE_RIGHT,
        FFTAdapterConfig,
        FPGAFFTReceiver,
    )
    from i2s_stream import AUTO_AUDIO_DEVICE, DEFAULT_CAPTURE_BACKEND, DEFAULT_CAPTURE_RATE_HZ, resolve_audio_device


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE") or AUTO_AUDIO_DEVICE
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "live_spectrogram_latest.png"


def _display_available() -> bool:
    return any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET"))


def _load_pyplot(requested_backend: str):
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for live spectrogram visualization. "
            "Install the project requirements first."
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


def _trim_history(
    fft_history: deque[np.ndarray],
    history_times: deque[float],
    now: float,
    history_seconds: float,
) -> None:
    cutoff = float(now) - max(0.0, float(history_seconds))
    while history_times and float(history_times[0]) < cutoff:
        history_times.popleft()
        fft_history.popleft()


def _prepare_history_array(fft_history: deque[np.ndarray]) -> np.ndarray:
    if not fft_history:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(list(fft_history), dtype=np.float32)


def _compute_fft_db(fft_cache: np.ndarray, rate: int, frame_bins: int, max_freq: float) -> tuple[np.ndarray, np.ndarray]:
    if fft_cache.ndim != 2 or fft_cache.shape[0] == 0 or fft_cache.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32)

    freq_per_bin = float(rate) / float(frame_bins)
    max_bin = min(fft_cache.shape[1], int(max_freq / freq_per_bin) + 1)
    fft_mag = np.maximum(fft_cache[:, :max_bin], 1e-9)
    fft_db = 20.0 * np.log10(fft_mag)
    freqs_hz = np.arange(max_bin, dtype=np.float32) * freq_per_bin
    return fft_db, freqs_hz


def _resolve_db_limits(
    fft_db: np.ndarray,
    *,
    min_db: Optional[float],
    max_db: Optional[float],
    dynamic_range_db: float,
) -> tuple[float, float]:
    if fft_db.size == 0:
        vmax = max_db if max_db is not None else 0.0
        vmin = min_db if min_db is not None else (vmax - dynamic_range_db)
        if vmin >= vmax:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    vmax = float(np.max(fft_db)) if max_db is None else float(max_db)
    vmin = float(vmax - dynamic_range_db) if min_db is None else float(min_db)
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _create_live_figure(plt):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [1.1, 2.2]},
    )
    spectrum_ax, spectrogram_ax = axes
    spectrum_line, = spectrum_ax.plot([], [], color="#cc5500", linewidth=1.8)
    spectrogram_im = spectrogram_ax.imshow(
        np.zeros((2, 2), dtype=np.float32),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
    )
    colorbar = fig.colorbar(spectrogram_im, ax=spectrogram_ax, pad=0.015)
    colorbar.set_label("Magnitude (dB)")
    return fig, spectrum_ax, spectrogram_ax, spectrum_line, spectrogram_im


def _update_live_figure(
    spectrum_ax,
    spectrogram_ax,
    spectrum_line,
    spectrogram_im,
    fft_cache: np.ndarray,
    history_times: np.ndarray,
    *,
    rate: int,
    frame_bins: int,
    max_freq: float,
    step_hz: float,
    history_seconds: float,
    dynamic_range_db: float,
    min_db: Optional[float],
    max_db: Optional[float],
    smoothed_frames: int,
) -> tuple[float, float]:
    fft_db, freqs_hz = _compute_fft_db(fft_cache, rate, frame_bins, max_freq)
    vmin, vmax = _resolve_db_limits(
        fft_db,
        min_db=min_db,
        max_db=max_db,
        dynamic_range_db=dynamic_range_db,
    )

    if fft_db.size == 0 or freqs_hz.size == 0:
        spectrum_line.set_data([], [])
        spectrogram_im.set_data(np.zeros((2, 2), dtype=np.float32))
        spectrogram_im.set_extent((-history_seconds, 0.0, 0.0, max_freq))
        spectrogram_im.set_clim(vmin=vmin, vmax=vmax)
        return 0.0, 0.0

    latest_db = fft_db[-1]
    peak_bin = int(np.argmax(latest_db))
    peak_freq_hz = float(freqs_hz[peak_bin]) if peak_bin < freqs_hz.size else 0.0
    freq_limit = float(freqs_hz[-1]) if freqs_hz.size else max_freq

    spectrum_line.set_data(freqs_hz, latest_db)
    spectrum_ax.set_xlim(0.0, freq_limit)
    spectrum_ax.set_ylim(vmin, vmax)
    spectrum_ax.set_ylabel("Magnitude (dB)")
    spectrum_ax.set_xlabel("Frequencia (Hz)")
    spectrum_ax.grid(True, which="both", alpha=0.25)
    spectrum_ax.set_title(
        f"Espectro ao vivo | pico {peak_freq_hz:.1f} Hz | media temporal {max(1, smoothed_frames)} frame(s)"
    )

    history_duration = float(history_times[-1] - history_times[0]) if history_times.size >= 2 else 0.0
    start_time = -history_duration
    spectrogram_im.set_data(fft_db.T)
    spectrogram_im.set_extent((start_time, 0.0, 0.0, freq_limit))
    spectrogram_im.set_clim(vmin=vmin, vmax=vmax)

    spectrogram_ax.set_xlim(-history_seconds, 0.0)
    spectrogram_ax.set_ylim(0.0, freq_limit)
    spectrogram_ax.set_title(f"Spectrograma ao vivo | janela temporal {history_seconds:.1f} s")
    spectrogram_ax.set_xlabel("Tempo relativo (s)")
    spectrogram_ax.set_ylabel("Frequencia (Hz)")

    tick_freqs_hz = np.arange(0.0, freq_limit + step_hz, step_hz)
    if tick_freqs_hz.size > 0:
        spectrogram_ax.set_yticks(tick_freqs_hz)
        spectrogram_ax.set_yticklabels([f"{int(freq)}" for freq in tick_freqs_hz])

    spectrogram_ax.set_xticks(np.linspace(-history_seconds, 0.0, 6))
    return peak_freq_hz, history_duration


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a rolling live spectrogram from the mirrored microphone I2S stream."
    )
    parser.add_argument("-D", "--device", default=DEFAULT_AUDIO_DEVICE, help="ALSA capture device")
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "arecord", "alsa-c"),
        default=DEFAULT_CAPTURE_BACKEND,
        help="Capture backend used to read the ALSA stream",
    )
    parser.add_argument("--capture-binary", default=None, help="Path to the native C capture helper")
    parser.add_argument("-r", "--rate", type=int, default=DEFAULT_CAPTURE_RATE_HZ, help="Sample rate in Hz")
    parser.add_argument("--frame-bins", type=int, default=512, help="Samples per FFT window")
    parser.add_argument("--useful-bins", type=int, default=256, help="Magnitude bins kept for visualization")
    parser.add_argument(
        "--channel-mode",
        choices=(CHANNEL_MODE_AUTO, CHANNEL_MODE_LEFT, CHANNEL_MODE_RIGHT, CHANNEL_MODE_AVERAGE),
        default=CHANNEL_MODE_AUTO,
        help="Channel used for the software FFT",
    )
    parser.add_argument("--sample-shift-bits", type=int, default=0, help="Arithmetic right shift before FFT")
    parser.add_argument("--keep-dc", action="store_true", help="Keep the DC component before FFT")
    parser.add_argument("--max-freq", type=float, default=20000.0, help="Maximum frequency shown on the plot")
    parser.add_argument("--step-hz", type=float, default=1000.0, help="Y-axis label spacing in Hz")
    parser.add_argument("--history-seconds", type=float, default=12.0, help="Rolling spectrogram history in seconds")
    parser.add_argument("--smooth-frames", type=int, default=4, help="Temporal averaging window applied before display")
    parser.add_argument("--fps", type=float, default=12.0, help="Maximum redraw rate")
    parser.add_argument("--dynamic-range-db", type=float, default=70.0, help="Automatic color range in dB")
    parser.add_argument("--min-db", type=float, default=None, help="Fixed minimum dB")
    parser.add_argument("--max-db", type=float, default=None, help="Fixed maximum dB")
    parser.add_argument("--backend", default="auto", help="Matplotlib backend or 'auto'")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="PNG output when headless")
    args = parser.parse_args(argv)

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.frame_bins <= 1:
        parser.error("--frame-bins must be greater than 1")
    if not 2 <= args.useful_bins <= (1 + args.frame_bins // 2):
        parser.error("--useful-bins must satisfy 2 <= useful-bins <= 1 + frame-bins/2")
    if args.sample_shift_bits < 0:
        parser.error("--sample-shift-bits must be non-negative")
    if args.max_freq <= 0 or args.step_hz <= 0 or args.history_seconds <= 0 or args.smooth_frames <= 0 or args.fps <= 0:
        parser.error("--max-freq, --step-hz, --history-seconds, --smooth-frames and --fps must be positive")
    if args.dynamic_range_db <= 0:
        parser.error("--dynamic-range-db must be positive")
    if args.min_db is not None and args.max_db is not None and args.min_db >= args.max_db:
        parser.error("--min-db must be smaller than --max-db")

    try:
        device = resolve_audio_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    cfg = FFTAdapterConfig(
        device=device,
        sample_rate=args.rate,
        frame_bins=args.frame_bins,
        useful_bins=args.useful_bins,
        capture_backend=args.capture_backend,
        capture_binary=args.capture_binary,
        channel_mode=args.channel_mode,
        sample_shift_bits=args.sample_shift_bits,
        remove_dc=not args.keep_dc,
    )

    plt, backend_name, interactive = _load_pyplot(args.backend)
    output_path = args.output_file.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, spectrum_ax, spectrogram_ax, spectrum_line, spectrogram_im = _create_live_figure(plt)
    fig.tight_layout()

    if interactive:
        plt.ion()
        plt.show(block=False)
        print(f"Using matplotlib backend: {backend_name}", flush=True)
    else:
        print(f"Using matplotlib backend: {backend_name} (headless mode, updating {output_path})", flush=True)

    fft_history: deque[np.ndarray] = deque()
    history_times: deque[float] = deque()
    smooth_history: deque[np.ndarray] = deque(maxlen=args.smooth_frames)
    history_lock = threading.Lock()
    capture_stop = threading.Event()
    capture_state = {
        "frame_counter": 0,
        "capture_error": None,
        "last_channel": CHANNEL_MODE_LEFT,
    }
    update_interval = 1.0 / float(args.fps)

    rx = FPGAFFTReceiver(cfg)
    try:
        rx.start()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

    print("Using ALSA capture device:", device, flush=True)
    print(
        "Reading mirrored microphone stream for live spectrogram:",
        f"channel_mode={cfg.channel_mode}",
        f"sample_shift_bits={cfg.sample_shift_bits}",
        f"remove_dc={cfg.remove_dc}",
        flush=True,
    )
    print("Close the plot window or press Ctrl+C to stop.", flush=True)

    last_render = time.monotonic()
    last_status = time.monotonic()
    peak_freq_hz = 0.0
    history_duration = 0.0

    def capture_loop() -> None:
        try:
            while not capture_stop.is_set():
                frame = rx.read_frame()
                if frame is None:
                    continue

                fft_bins, _ = frame
                fft_frame = np.asarray(fft_bins, dtype=np.float32)
                now = time.monotonic()

                with history_lock:
                    smooth_history.append(fft_frame)
                    if args.smooth_frames == 1:
                        display_frame = fft_frame
                    else:
                        display_frame = np.mean(np.asarray(smooth_history, dtype=np.float32), axis=0, dtype=np.float32)

                    fft_history.append(display_frame.copy())
                    history_times.append(now)
                    _trim_history(fft_history, history_times, now, args.history_seconds)
                    capture_state["frame_counter"] = int(capture_state["frame_counter"]) + 1
                    capture_state["last_channel"] = str(rx.last_channel_used)
        except Exception as exc:  # pragma: no cover - depends on live device state
            capture_state["capture_error"] = str(exc)

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    try:
        while True:
            if interactive and not plt.fignum_exists(fig.number):
                break

            now = time.monotonic()
            if capture_state["capture_error"] is not None:
                raise RuntimeError(str(capture_state["capture_error"]))

            if (now - last_render) < update_interval:
                if interactive:
                    plt.pause(0.001)
                else:
                    time.sleep(min(0.01, update_interval))
                continue

            with history_lock:
                fft_cache = _prepare_history_array(fft_history)
                history_time_array = np.asarray(history_times, dtype=np.float64)
                frame_counter = int(capture_state["frame_counter"])
                last_channel = str(capture_state["last_channel"])

            peak_freq_hz, history_duration = _update_live_figure(
                spectrum_ax,
                spectrogram_ax,
                spectrum_line,
                spectrogram_im,
                fft_cache,
                history_time_array,
                rate=args.rate,
                frame_bins=args.frame_bins,
                max_freq=args.max_freq,
                step_hz=args.step_hz,
                history_seconds=args.history_seconds,
                dynamic_range_db=args.dynamic_range_db,
                min_db=args.min_db,
                max_db=args.max_db,
                smoothed_frames=args.smooth_frames,
            )

            if interactive:
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
            else:
                fig.savefig(output_path, dpi=120)

            last_render = now

            if now - last_status >= 1.0:
                print(
                    "live:",
                    f"frames={frame_counter}",
                    f"history={history_duration:.2f}s",
                    f"channel={last_channel}",
                    f"peak_freq_hz={peak_freq_hz:.1f}",
                    flush=True,
                )
                if not interactive:
                    print("PNG atualizada:", output_path, flush=True)
                last_status = now

    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        capture_stop.set()
        rx.stop()
        capture_thread.join(timeout=1.0)

    if not interactive:
        fig.savefig(output_path, dpi=120)
        print("PNG final:", output_path, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
