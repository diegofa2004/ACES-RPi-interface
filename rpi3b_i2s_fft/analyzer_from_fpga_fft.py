import argparse
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    from .compararEvento import compararEvento
    from .fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
    from .i2s_stream import AUTO_AUDIO_DEVICE, resolve_audio_device
except ImportError:
    from compararEvento import compararEvento
    from fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
    from i2s_stream import AUTO_AUDIO_DEVICE, resolve_audio_device


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE") or AUTO_AUDIO_DEVICE
WORK_DIR = Path(__file__).resolve().parent
EVENTO_FILENAME = WORK_DIR / "evento.npy"
FFT_FILENAME = WORK_DIR / "fft.npy"
EVENTO_TMP_FILENAME = WORK_DIR / "evento_tmp.npy"
FFT_TMP_FILENAME = WORK_DIR / "fft_tmp.npy"

PREBUFFER_SECONDS = 5.0
HISTORY_SECONDS = 15.0
RECORD_SECONDS = 5.0


def save_event_snapshot(evento: np.ndarray, fft: np.ndarray) -> None:
    np.save(EVENTO_TMP_FILENAME, evento)
    os.replace(EVENTO_TMP_FILENAME, EVENTO_FILENAME)
    np.save(FFT_TMP_FILENAME, fft)
    os.replace(FFT_TMP_FILENAME, FFT_FILENAME)


def frames_for_seconds(sample_rate: int, frame_bins: int, seconds: float) -> int:
    frames_per_second = sample_rate / frame_bins
    return max(1, int(math.ceil(frames_per_second * seconds)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feed circular buffers from FPGA I2S FFT stream using the same event logic as pyserial."
    )
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_AUDIO_DEVICE,
        help="ALSA capture device (default: $AUDIO_DEVICE if set, otherwise auto-detect)",
    )
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

    try:
        device = resolve_audio_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    os.chdir(WORK_DIR)

    buffer_size = frames_for_seconds(args.rate, args.frame_bins, PREBUFFER_SECONDS)
    buffer_size2 = frames_for_seconds(args.rate, args.frame_bins, HISTORY_SECONDS)
    buffer_size3 = frames_for_seconds(args.rate, args.frame_bins, PREBUFFER_SECONDS)
    buffer_size4 = frames_for_seconds(args.rate, args.frame_bins, HISTORY_SECONDS)

    lock = threading.Lock()
    buffer = deque(maxlen=buffer_size)
    buffer2 = deque(maxlen=buffer_size2)
    buffer3 = deque(maxlen=buffer_size3)
    buffer4 = deque(maxlen=buffer_size4)

    state = {
        "recording": False,
        "record_start": 0.0,
        "future_buffer": [],
        "future_buffer2": [],
        "last_event_time": 0.0,
    }

    def toggle_recording() -> None:
        while True:
            try:
                input()
            except EOFError:
                return

            with lock:
                if state["recording"]:
                    continue

                state["future_buffer"] = []
                state["future_buffer2"] = []
                state["recording"] = True
                state["record_start"] = time.time()

            print("Gravando evento com pre-buffer de 5 s...", flush=True)

    try:
        cfg = FFTAdapterConfig(
            device=device,
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

    threading.Thread(target=toggle_recording, daemon=True).start()
    threading.Thread(
        target=compararEvento,
        args=(buffer2, buffer4, lock, lambda: state["last_event_time"]),
        daemon=True,
    ).start()

    print("Using ALSA capture device:", device, flush=True)
    print("Reading FPGA FFT stream from I2S...", flush=True)
    print(
        "Buffer sizes:",
        f"pre_mfcc={buffer_size}",
        f"history_mfcc={buffer_size2}",
        f"pre_fft={buffer_size3}",
        f"history_fft={buffer_size4}",
        flush=True,
    )
    print("Press ENTER to save an event like the pyserial flow.", flush=True)
    print("Comparison starts after a reference event is saved and the 15 s cooldown ends.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            frame = rx.read_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            fft_bins, mfcc = frame
            mfcc8 = np.asarray(mfcc[:8], dtype=np.float32)
            fft_bins = np.asarray(fft_bins, dtype=np.float32)
            complete_event = None

            with lock:
                buffer.append(mfcc8.copy())
                buffer2.append(mfcc8.copy())
                buffer3.append(fft_bins.copy())
                buffer4.append(fft_bins.copy())

                if state["recording"]:
                    state["future_buffer"].append(mfcc8.copy())
                    state["future_buffer2"].append(fft_bins.copy())

                    if time.time() - state["record_start"] >= RECORD_SECONDS:
                        evento = np.array(list(buffer) + state["future_buffer"], dtype=np.float32)
                        fft = np.array(list(buffer3) + state["future_buffer2"], dtype=np.float32)
                        state["last_event_time"] = time.time()
                        state["recording"] = False
                        complete_event = (evento, fft)

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
