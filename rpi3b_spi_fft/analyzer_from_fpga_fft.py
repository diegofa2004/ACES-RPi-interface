import argparse
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from .compararEvento import DirectComparatorConfig, compararEvento
    from .fpga_audio_adapter import AudioCaptureConfig, FPGAAudioReceiver
    from .i2s_stream import AUTO_AUDIO_DEVICE, DEFAULT_CAPTURE_BACKEND, DEFAULT_CAPTURE_RATE_HZ, resolve_audio_device
except ImportError:
    from compararEvento import DirectComparatorConfig, compararEvento
    from fpga_audio_adapter import AudioCaptureConfig, FPGAAudioReceiver
    from i2s_stream import AUTO_AUDIO_DEVICE, DEFAULT_CAPTURE_BACKEND, DEFAULT_CAPTURE_RATE_HZ, resolve_audio_device


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE") or AUTO_AUDIO_DEVICE
WORK_DIR = Path(__file__).resolve().parent
EVENTO_FILENAME = WORK_DIR / "evento.npy"
FFT_FILENAME = WORK_DIR / "fft.npy"
EVENTO_TMP_FILENAME = WORK_DIR / "evento_tmp.npy"
FFT_TMP_FILENAME = WORK_DIR / "fft_tmp.npy"
RECORD_TRIGGER_FILENAME = WORK_DIR / "record_button.trigger"

PREBUFFER_SECONDS = 5.0
HISTORY_SECONDS = 15.0
RECORD_SECONDS = 5.0


def save_event_snapshot(evento: np.ndarray, fft: np.ndarray) -> None:
    np.save(EVENTO_TMP_FILENAME, evento)
    os.replace(EVENTO_TMP_FILENAME, EVENTO_FILENAME)
    np.save(FFT_TMP_FILENAME, fft)
    os.replace(FFT_TMP_FILENAME, FFT_FILENAME)


def frames_for_seconds(sample_rate: int, frame_samples: int, seconds: float) -> int:
    frames_per_second = sample_rate / frame_samples
    return max(1, int(math.ceil(frames_per_second * seconds)))


def create_analysis_buffers(
    sample_rate: int,
    frame_samples: int,
    *,
    prebuffer_seconds: float = PREBUFFER_SECONDS,
    history_seconds: float = HISTORY_SECONDS,
) -> dict[str, object]:
    _ = frames_for_seconds(sample_rate, frame_samples, prebuffer_seconds)
    _ = frames_for_seconds(sample_rate, frame_samples, history_seconds)
    return {
        "pre_mfcc": deque(),
        "history_mfcc": deque(),
        "pre_fft": deque(),
        "history_fft": deque(),
        "pre_times": deque(),
        "history_times": deque(),
        "pre_window_seconds": float(prebuffer_seconds),
        "history_window_seconds": float(history_seconds),
    }


def create_runtime_state() -> dict[str, object]:
    return {
        "recording": False,
        "record_start": 0.0,
        "future_mfcc": [],
        "future_fft": [],
        "captured_pre_mfcc": [],
        "captured_pre_fft": [],
        "last_event_time": 0.0,
    }


def arm_recording(state: dict[str, object], now: float, buffers: Optional[dict[str, object]] = None) -> bool:
    if bool(state["recording"]):
        return False

    state["future_mfcc"] = []
    state["future_fft"] = []
    state["captured_pre_mfcc"] = list(buffers["pre_mfcc"]) if buffers is not None else []
    state["captured_pre_fft"] = list(buffers["pre_fft"]) if buffers is not None else []
    state["recording"] = True
    state["record_start"] = float(now)
    return True


def _trim_timed_buffer(
    mfcc_buffer: deque,
    fft_buffer: deque,
    time_buffer: deque,
    now: float,
    window_seconds: float,
) -> None:
    cutoff = float(now) - max(0.0, float(window_seconds))
    while time_buffer and float(time_buffer[0]) < cutoff:
        time_buffer.popleft()
        mfcc_buffer.popleft()
        fft_buffer.popleft()


def ingest_frame(
    buffers: dict[str, object],
    state: dict[str, object],
    mfcc: np.ndarray,
    fft_bins: np.ndarray,
    now: float,
    *,
    record_seconds: float = RECORD_SECONDS,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mfcc8 = np.asarray(mfcc[:8], dtype=np.float32)
    fft_frame = np.asarray(fft_bins, dtype=np.float32)

    buffers["pre_mfcc"].append(mfcc8.copy())
    buffers["history_mfcc"].append(mfcc8.copy())
    buffers["pre_fft"].append(fft_frame.copy())
    buffers["history_fft"].append(fft_frame.copy())
    buffers["pre_times"].append(float(now))
    buffers["history_times"].append(float(now))

    _trim_timed_buffer(
        buffers["pre_mfcc"],
        buffers["pre_fft"],
        buffers["pre_times"],
        now,
        float(buffers["pre_window_seconds"]),
    )
    _trim_timed_buffer(
        buffers["history_mfcc"],
        buffers["history_fft"],
        buffers["history_times"],
        now,
        float(buffers["history_window_seconds"]),
    )

    if not bool(state["recording"]):
        return None

    future_mfcc = state["future_mfcc"]
    future_fft = state["future_fft"]
    assert isinstance(future_mfcc, list)
    assert isinstance(future_fft, list)

    future_mfcc.append(mfcc8.copy())
    future_fft.append(fft_frame.copy())

    if float(now) - float(state["record_start"]) < record_seconds:
        return None

    captured_pre_mfcc = state["captured_pre_mfcc"]
    captured_pre_fft = state["captured_pre_fft"]
    assert isinstance(captured_pre_mfcc, list)
    assert isinstance(captured_pre_fft, list)

    evento = np.asarray(captured_pre_mfcc + future_mfcc, dtype=np.float32)
    fft = np.asarray(captured_pre_fft + future_fft, dtype=np.float32)

    state["last_event_time"] = float(now)
    state["recording"] = False
    state["future_mfcc"] = []
    state["future_fft"] = []
    return evento, fft


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture raw audio forwarded by the FPGA over I2S/ALSA, compute FFT+MFCC on the Raspberry Pi, and keep the same event-comparison flow."
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
        help="Audio capture backend used to read the ALSA stream",
    )
    parser.add_argument(
        "--capture-binary",
        default=None,
        help="Path to the compiled native ALSA helper when --capture-backend=alsa-c",
    )
    parser.add_argument("-r", "--rate", type=int, default=DEFAULT_CAPTURE_RATE_HZ, help="Host-side sample rate in Hz")
    parser.add_argument("--frame-bins", type=int, default=512, help="PCM samples per analysis frame and FFT size")
    parser.add_argument("--useful-bins", type=int, default=256, help="Positive-frequency FFT bins kept for similarity")
    parser.add_argument("--read-frames", type=int, default=512, help="Read quantum requested from the ALSA backend")
    parser.add_argument(
        "--sample-shift-bits",
        type=int,
        default=8,
        help="Arithmetic right shift applied to each captured S32_LE sample to recover the 24-bit microphone payload",
    )
    parser.add_argument(
        "--mono-channel",
        choices=("auto", "left", "right", "average"),
        default="auto",
        help="How to collapse the captured stereo stream into the mono microphone signal used for FFT/MFCC",
    )
    parser.add_argument(
        "--compare-threshold",
        type=float,
        default=DirectComparatorConfig.absolute_threshold,
        help="Absolute similarity threshold for the direct comparator",
    )
    parser.add_argument(
        "--compare-margin",
        type=float,
        default=DirectComparatorConfig.margin_threshold,
        help="Minimum score margin above the recent baseline for the direct comparator",
    )
    parser.add_argument(
        "--compare-poll-ms",
        type=float,
        default=DirectComparatorConfig.poll_interval_seconds * 1000.0,
        help="Polling interval of the direct comparator in milliseconds",
    )
    parser.add_argument(
        "--compare-min-energy-ratio",
        type=float,
        default=DirectComparatorConfig.min_energy_ratio,
        help="Minimum energy ratio accepted for candidate windows",
    )
    parser.add_argument(
        "--compare-max-reference-frames",
        type=int,
        default=DirectComparatorConfig.max_reference_frames,
        help="Maximum number of FFT frames kept from the recorded reference event",
    )
    parser.add_argument(
        "--compare-search-frames",
        type=int,
        default=DirectComparatorConfig.max_search_frames,
        help="Maximum number of recent FFT frames searched for a match",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.frame_bins <= 0:
        parser.error("--frame-bins must be positive")
    if not 2 <= args.useful_bins <= ((args.frame_bins // 2) + 1):
        parser.error("--useful-bins must satisfy 2 <= useful-bins <= (frame-bins // 2) + 1")
    if args.read_frames <= 0:
        parser.error("--read-frames must be positive")
    if not 0 <= args.sample_shift_bits <= 16:
        parser.error("--sample-shift-bits must be between 0 and 16")
    if args.capture_binary is not None and not Path(args.capture_binary).expanduser().exists():
        parser.error(f"--capture-binary does not exist: {args.capture_binary}")
    if not 0.0 < args.compare_threshold <= 1.0:
        parser.error("--compare-threshold must satisfy 0 < compare-threshold <= 1")
    if args.compare_margin < 0.0:
        parser.error("--compare-margin must be non-negative")
    if args.compare_poll_ms <= 0.0:
        parser.error("--compare-poll-ms must be positive")
    if args.compare_min_energy_ratio <= 0.0:
        parser.error("--compare-min-energy-ratio must be positive")
    if args.compare_max_reference_frames <= 0:
        parser.error("--compare-max-reference-frames must be positive")
    if args.compare_search_frames <= 0:
        parser.error("--compare-search-frames must be positive")

    device = resolve_audio_device(args.device)
    os.chdir(WORK_DIR)

    try:
        cfg = AudioCaptureConfig(
            device=device,
            sample_rate=args.rate,
            frame_length=args.frame_bins,
            useful_bins=args.useful_bins,
            capture_backend=args.capture_backend,
            capture_binary=args.capture_binary,
            read_frames=args.read_frames,
            sample_shift_bits=args.sample_shift_bits,
            mono_channel=args.mono_channel,
        )
    except ValueError as exc:
        parser.error(str(exc))

    buffers = create_analysis_buffers(args.rate, args.frame_bins)
    state = create_runtime_state()
    lock = threading.Lock()
    compare_config = DirectComparatorConfig(
        absolute_threshold=args.compare_threshold,
        margin_threshold=args.compare_margin,
        poll_interval_seconds=args.compare_poll_ms / 1000.0,
        min_energy_ratio=args.compare_min_energy_ratio,
        max_reference_frames=args.compare_max_reference_frames,
        max_search_frames=args.compare_search_frames,
    )

    def trigger_recording(source: str) -> bool:
        with lock:
            armed = arm_recording(state, time.time(), buffers)
            if not armed:
                return False
        print(f"Gravando evento com pre-buffer de 5 s... fonte={source}", flush=True)
        return True

    def toggle_recording() -> None:
        while True:
            try:
                input()
            except EOFError:
                return
            trigger_recording("stdin_enter")

    def watch_record_trigger() -> None:
        last_mtime_ns = RECORD_TRIGGER_FILENAME.stat().st_mtime_ns if RECORD_TRIGGER_FILENAME.exists() else 0
        while True:
            try:
                stat_result = RECORD_TRIGGER_FILENAME.stat()
            except FileNotFoundError:
                time.sleep(0.10)
                continue

            current_mtime_ns = stat_result.st_mtime_ns
            if current_mtime_ns != last_mtime_ns:
                last_mtime_ns = current_mtime_ns
                trigger_recording("gpio_button")
            time.sleep(0.10)

    rx = FPGAAudioReceiver(cfg)
    try:
        rx.start()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

    threading.Thread(target=toggle_recording, daemon=True).start()
    threading.Thread(target=watch_record_trigger, daemon=True).start()
    threading.Thread(
        target=compararEvento,
        args=(buffers["history_mfcc"], buffers["history_fft"], lock, lambda: state["last_event_time"], compare_config),
        daemon=True,
    ).start()

    print("Using ALSA device:", device, flush=True)
    print(
        "Reading raw microphone audio from FPGA over I2S/ALSA...",
        f"backend={cfg.capture_backend}",
        f"frame_samples={cfg.frame_length}",
        f"useful_bins={cfg.useful_bins}",
        f"sample_shift_bits={cfg.sample_shift_bits}",
        f"mono_channel={cfg.mono_channel}",
        flush=True,
    )
    print("Press ENTER to save an event like the old pyserial flow.", flush=True)
    print(f"External record trigger file: {RECORD_TRIGGER_FILENAME}", flush=True)
    print("Comparison starts after a reference event is saved and the 15 s cooldown ends.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            frame = rx.read_frame()
            if frame is None:
                continue

            fft_bins, mfcc = frame
            now = time.time()

            with lock:
                complete_event = ingest_frame(buffers, state, mfcc, fft_bins, now)

            if complete_event is not None:
                evento, fft = complete_event
                save_event_snapshot(evento, fft)
                print("evento de 10s salvo", flush=True)
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        rx.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
