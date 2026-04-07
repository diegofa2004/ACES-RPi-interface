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
        DEFAULT_BFPEXP_HOLD_PAIRS,
        DEFAULT_TAG_LOSS_TOLERANCE_PAIRS,
        FFTAdapterConfig,
        FPGAFFTReceiver,
    )
    from .i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_BACKEND,
        DEFAULT_CAPTURE_RATE_HZ,
        DEFAULT_FFT_PACKET_INDEX_BASE,
        DEFAULT_PACKET_INDEX_BITS,
        DEFAULT_PACKET_INDEX_SHIFT,
        DEFAULT_PAYLOAD_BITS,
        DEFAULT_TAG_BFPEXP,
        DEFAULT_TAG_FFT,
        DEFAULT_TAG_IDLE,
        DEFAULT_TAG_MASK,
        DEFAULT_TAG_SHIFT,
        resolve_audio_device,
    )
except ImportError:
    from fpga_fft_adapter import (
        DEFAULT_BFPEXP_HOLD_PAIRS,
        DEFAULT_TAG_LOSS_TOLERANCE_PAIRS,
        FFTAdapterConfig,
        FPGAFFTReceiver,
    )
    from i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_BACKEND,
        DEFAULT_CAPTURE_RATE_HZ,
        DEFAULT_FFT_PACKET_INDEX_BASE,
        DEFAULT_PACKET_INDEX_BITS,
        DEFAULT_PACKET_INDEX_SHIFT,
        DEFAULT_PAYLOAD_BITS,
        DEFAULT_TAG_BFPEXP,
        DEFAULT_TAG_FFT,
        DEFAULT_TAG_IDLE,
        DEFAULT_TAG_MASK,
        DEFAULT_TAG_SHIFT,
        resolve_audio_device,
    )


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


def _resolve_sync_cli_defaults(args: argparse.Namespace) -> dict[str, object]:
    preset = getattr(args, "sync_mode", None) or getattr(args, "sync_preset", None)

    if args.use_i2s_tags is not None:
        use_i2s_tags = args.use_i2s_tags
    else:
        use_i2s_tags = True

    if args.bfpexp_hold_pairs is not None:
        bfpexp_hold_pairs = args.bfpexp_hold_pairs
    else:
        bfpexp_hold_pairs = DEFAULT_BFPEXP_HOLD_PAIRS

    if args.loss_tolerance_pairs is not None:
        loss_tolerance_pairs = args.loss_tolerance_pairs
    else:
        loss_tolerance_pairs = DEFAULT_TAG_LOSS_TOLERANCE_PAIRS

    if args.allow_fft_without_bfpexp is not None:
        allow_fft_without_bfpexp = args.allow_fft_without_bfpexp
    else:
        allow_fft_without_bfpexp = preset == "tolerant"

    if preset == "strict":
        sync_mode = "strict"
    elif preset == "tolerant":
        sync_mode = "tolerant"
    elif allow_fft_without_bfpexp:
        sync_mode = "tolerant"
    else:
        sync_mode = "strict"

    return {
        "sync_mode": sync_mode,
        "use_i2s_tags": use_i2s_tags,
        "bfpexp_hold_pairs": bfpexp_hold_pairs,
        "loss_tolerance_pairs": loss_tolerance_pairs,
        "allow_fft_without_bfpexp": allow_fft_without_bfpexp,
    }


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


def _compute_fft_db(fft_cache: np.ndarray, rate: int, frame_bins: int, max_freq: float) -> tuple[np.ndarray, np.ndarray, float]:
    if fft_cache.ndim != 2 or fft_cache.shape[0] == 0 or fft_cache.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32), float(rate) / float(frame_bins)

    freq_per_bin = float(rate) / float(frame_bins)
    max_bin = min(fft_cache.shape[1], int(max_freq / freq_per_bin) + 1)
    fft_mag = np.maximum(fft_cache[:, :max_bin], 1e-9)
    fft_db = 20.0 * np.log10(fft_mag)
    freqs_hz = np.arange(max_bin, dtype=np.float32) * freq_per_bin
    return fft_db, freqs_hz, freq_per_bin


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
    fft_db, freqs_hz, freq_per_bin = _compute_fft_db(fft_cache, rate, frame_bins, max_freq)
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

    if history_times.size >= 2:
        history_duration = float(history_times[-1] - history_times[0])
    else:
        history_duration = 0.0
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

    if history_duration <= 0.0:
        history_duration = max(1e-3, float(frame_bins) / float(rate))

    xtick_count = 6
    tick_start = -history_seconds
    tick_stop = 0.0
    spectrogram_ax.set_xticks(np.linspace(tick_start, tick_stop, xtick_count))

    return peak_freq_hz, history_duration


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a rolling live spectrogram directly from the FPGA I2S FFT stream."
    )
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_AUDIO_DEVICE,
        help="ALSA capture device (default: $AUDIO_DEVICE if set, otherwise auto-detect)",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "arecord", "alsa-c"),
        default=DEFAULT_CAPTURE_BACKEND,
        help="Capture backend used to read the ALSA stream",
    )
    parser.add_argument(
        "--capture-binary",
        default=None,
        help="Path to the compiled native C capture helper (used when --capture-backend=alsa-c)",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=DEFAULT_CAPTURE_RATE_HZ,
        help="Host-side ALSA sample rate in Hz (nominal wire rate is 48828.125 Hz)",
    )
    parser.add_argument("--frame-bins", type=int, default=512, help="Complex bins per FPGA FFT frame")
    parser.add_argument("--useful-bins", type=int, default=256, help="Bins kept for visualization")
    parser.add_argument("--max-freq", type=float, default=20000.0, help="Maximum frequency shown on the plot")
    parser.add_argument("--step-hz", type=float, default=1000.0, help="Y-axis label spacing in Hz")
    parser.add_argument("--history-seconds", type=float, default=12.0, help="Rolling spectrogram history in seconds")
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=4,
        help="Temporal averaging window applied before display (1 keeps full frame-to-frame sharpness)",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Maximum redraw rate of the GUI/PNG output")
    parser.add_argument(
        "--dynamic-range-db",
        type=float,
        default=70.0,
        help="Automatic spectrogram color range in dB when --min-db is not specified",
    )
    parser.add_argument("--min-db", type=float, default=None, help="Fixed minimum dB shown on the plots")
    parser.add_argument("--max-db", type=float, default=None, help="Fixed maximum dB shown on the plots")
    parser.add_argument(
        "--backend",
        default="auto",
        help="Matplotlib backend. Use 'auto' for GUI when available and 'Agg' when headless.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="PNG output path used when running in headless mode",
    )
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
        help="Output GPIO line number: pulsed when a FFT frame is consumed",
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
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument(
        "--sync-mode",
        choices=("strict", "tolerant"),
        default=None,
        help="Advanced selector for tagged-stream sync behavior",
    )
    sync_group.add_argument(
        "--strict-sync",
        dest="sync_preset",
        action="store_const",
        const="strict",
        help="Convenience preset: tagged mode with full BFPEXP preamble required",
    )
    sync_group.add_argument(
        "--tolerant-sync",
        dest="sync_preset",
        action="store_const",
        const="tolerant",
        help="Convenience preset: allows FFT sync without BFPEXP preamble",
    )
    parser.add_argument(
        "--use-i2s-tags",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode per-word in-band tags (idle/BFPEXP/FFT) from I2S stream (default: enabled)",
    )
    parser.add_argument("--packet-index-shift", type=int, default=DEFAULT_PACKET_INDEX_SHIFT, help="Bit shift of the packet-index field")
    parser.add_argument("--packet-index-bits", type=int, default=DEFAULT_PACKET_INDEX_BITS, help="Packet-index field width in bits")
    parser.add_argument("--fft-packet-index-base", type=int, default=DEFAULT_FFT_PACKET_INDEX_BASE, help="First packet index used by FFT payload words")
    parser.add_argument("--tag-shift", type=int, default=DEFAULT_TAG_SHIFT, help="Bit shift of type tag in each 32-bit word")
    parser.add_argument("--tag-mask", type=lambda v: int(v, 0), default=DEFAULT_TAG_MASK, help="Bitmask for type tag")
    parser.add_argument("--payload-bits", type=int, default=DEFAULT_PAYLOAD_BITS, help="Signed payload width inside each word")
    parser.add_argument("--tag-idle", type=int, default=DEFAULT_TAG_IDLE, help="Tag value representing idle/no data")
    parser.add_argument("--tag-bfpexp", type=int, default=DEFAULT_TAG_BFPEXP, help="Tag value representing BFPEXP data")
    parser.add_argument("--tag-fft", type=int, default=DEFAULT_TAG_FFT, help="Tag value representing FFT complex bins")
    parser.add_argument(
        "--bfpexp-hold-pairs",
        type=int,
        default=None,
        help="Required consecutive BFPEXP-tagged stereo pairs before a new FFT burst is accepted",
    )
    parser.add_argument(
        "--loss-tolerance-pairs",
        type=int,
        default=None,
        help="Tolerated corrupted/missing tagged stereo pairs per BFPEXP preamble or FFT burst",
    )
    parser.add_argument(
        "--allow-fft-without-bfpexp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Accept FFT-tagged frame start even if no BFPEXP tag was observed first",
    )
    parser.add_argument(
        "--apply-bfpexp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply the FFT block-floating exponent before plotting magnitudes (default: enabled)",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.frame_bins <= 0:
        parser.error("--frame-bins must be positive")
    if args.frame_bins > args.fft_packet_index_base:
        parser.error("--frame-bins must fit inside the FFT packet-index range")
    if not 2 <= args.useful_bins <= args.frame_bins:
        parser.error("--useful-bins must satisfy 2 <= useful-bins <= frame-bins")
    if args.max_freq <= 0:
        parser.error("--max-freq must be positive")
    if args.step_hz <= 0:
        parser.error("--step-hz must be positive")
    if args.history_seconds <= 0:
        parser.error("--history-seconds must be positive")
    if args.smooth_frames <= 0:
        parser.error("--smooth-frames must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.dynamic_range_db <= 0:
        parser.error("--dynamic-range-db must be positive")
    if args.payload_bits <= 0:
        parser.error("--payload-bits must be positive")
    if args.packet_index_bits <= 0:
        parser.error("--packet-index-bits must be positive")
    if args.min_db is not None and args.max_db is not None and args.min_db >= args.max_db:
        parser.error("--min-db must be smaller than --max-db")

    sync_cfg = _resolve_sync_cli_defaults(args)
    use_i2s_tags = bool(sync_cfg["use_i2s_tags"])
    bfpexp_hold_pairs = int(sync_cfg["bfpexp_hold_pairs"])
    loss_tolerance_pairs = int(sync_cfg["loss_tolerance_pairs"])
    allow_fft_without_bfpexp = bool(sync_cfg["allow_fft_without_bfpexp"])
    apply_bfpexp = True if args.apply_bfpexp is None else bool(args.apply_bfpexp)
    sync_mode = str(sync_cfg["sync_mode"])

    if bfpexp_hold_pairs <= 0:
        parser.error("--bfpexp-hold-pairs must be positive")
    if loss_tolerance_pairs < 0:
        parser.error("--loss-tolerance-pairs must be non-negative")

    try:
        device = resolve_audio_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    try:
        cfg = FFTAdapterConfig(
            device=device,
            sample_rate=args.rate,
            frame_bins=args.frame_bins,
            useful_bins=args.useful_bins,
            capture_backend=args.capture_backend,
            capture_binary=args.capture_binary,
            gpio_chip=args.gpio_chip,
            bfpexp_flag_line=args.bfpexp_flag_line,
            done_line=args.done_line,
            flag_active_high=not args.flag_active_low,
            done_pulse_seconds=max(0.0, args.done_pulse_ms / 1000.0),
            handshake_timeout_seconds=max(0.001, args.handshake_timeout_ms / 1000.0),
            wait_for_flag_falling_edge=not args.wait_low_level,
            use_i2s_tags=use_i2s_tags,
            packet_index_shift=args.packet_index_shift,
            packet_index_bits=args.packet_index_bits,
            fft_packet_index_base=args.fft_packet_index_base,
            tag_shift=args.tag_shift,
            tag_mask=args.tag_mask,
            payload_bits=args.payload_bits,
            tag_idle=args.tag_idle,
            tag_bfpexp=args.tag_bfpexp,
            tag_fft=args.tag_fft,
            apply_bfpexp=apply_bfpexp,
            require_bfpexp_before_fft=not allow_fft_without_bfpexp,
            bfpexp_pairs_required=bfpexp_hold_pairs,
            loss_tolerance_pairs=loss_tolerance_pairs,
        )
    except ValueError as exc:
        parser.error(str(exc))

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
        "last_bfpexp": 0,
    }
    update_interval = 1.0 / float(args.fps)
    freq_resolution_hz = float(args.rate) / float(args.frame_bins)
    nominal_frames_per_second = float(args.rate) / float(args.frame_bins)
    nominal_time_resolution_ms = 1000.0 / nominal_frames_per_second

    rx = FPGAFFTReceiver(cfg)
    try:
        rx.start()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

    print("Using ALSA capture device:", device, flush=True)
    print("Reading FPGA FFT stream from I2S for live spectrogram...", flush=True)
    if cfg.use_i2s_tags:
        print(f"Sync preset: {sync_mode}", flush=True)
        print(
            f"Tagged mode: expecting up to {cfg.bfpexp_pairs_required} BFPEXP pairs before FFT frame start.",
            flush=True,
        )
        if cfg.loss_tolerance_pairs > 0:
            print(
                f"Tagged mode robustness: allowing up to {cfg.loss_tolerance_pairs} corrupted/missing pairs inside each frame.",
                flush=True,
            )
    print(
        "Display resolution (nominal):",
        f"freq_bin={freq_resolution_hz:.3f} Hz",
        f"time_frame={nominal_time_resolution_ms:.3f} ms",
        "history_trim=wall_clock",
        flush=True,
    )
    print(
        "Viewer tuning:",
        f"history_seconds={args.history_seconds:.2f}",
        f"smooth_frames={args.smooth_frames}",
        f"fps={args.fps:.2f}",
        f"apply_bfpexp={cfg.apply_bfpexp}",
        flush=True,
    )
    print("Close the plot window or press Ctrl+C to stop.", flush=True)

    render_counter = 0
    last_render = time.monotonic()
    last_status = time.monotonic()
    peak_freq_hz = 0.0
    history_duration = 0.0
    effective_fps = 0.0

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
                    capture_state["last_bfpexp"] = int(rx.last_frame_bfpexp)
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
                last_bfpexp = int(capture_state["last_bfpexp"])

            if history_time_array.size >= 2:
                elapsed = float(history_time_array[-1] - history_time_array[0])
                effective_fps = float((history_time_array.size - 1) / elapsed) if elapsed > 0.0 else 0.0
            else:
                effective_fps = 0.0

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
            render_counter += 1

            if now - last_status >= 1.0:
                print(
                    "live:",
                    f"frames={frame_counter}",
                    f"renders={render_counter}",
                    f"history={history_duration:.2f}s",
                    f"effective_fps={effective_fps:.1f}",
                    f"bfpexp={last_bfpexp}",
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
