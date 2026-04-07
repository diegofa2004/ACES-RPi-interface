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
    from .fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
    from .i2s_stream import AUTO_AUDIO_DEVICE, DEFAULT_CAPTURE_BACKEND, DEFAULT_CAPTURE_RATE_HZ
except ImportError:
    from compararEvento import DirectComparatorConfig, compararEvento
    from fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
    from i2s_stream import AUTO_AUDIO_DEVICE, DEFAULT_CAPTURE_BACKEND, DEFAULT_CAPTURE_RATE_HZ


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


def create_analysis_buffers(
    sample_rate: int,
    frame_bins: int,
    *,
    prebuffer_seconds: float = PREBUFFER_SECONDS,
    history_seconds: float = HISTORY_SECONDS,
) -> dict[str, object]:
    del sample_rate, frame_bins
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
        "future_buffer": [],
        "future_buffer2": [],
        "captured_prebuffer": [],
        "captured_prebuffer2": [],
        "last_event_time": 0.0,
    }


def arm_recording(state: dict[str, object], now: float, buffers: Optional[dict[str, object]] = None) -> bool:
    if bool(state["recording"]):
        return False

    state["future_buffer"] = []
    state["future_buffer2"] = []
    state["captured_prebuffer"] = list(buffers["pre_mfcc"]) if buffers is not None else []
    state["captured_prebuffer2"] = list(buffers["pre_fft"]) if buffers is not None else []
    state["recording"] = True
    state["record_start"] = now
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

    future_mfcc = state["future_buffer"]
    future_fft = state["future_buffer2"]
    assert isinstance(future_mfcc, list)
    assert isinstance(future_fft, list)

    future_mfcc.append(mfcc8.copy())
    future_fft.append(fft_frame.copy())

    record_start = float(state["record_start"])
    if now - record_start < record_seconds:
        return None

    captured_pre_mfcc = state["captured_prebuffer"]
    captured_pre_fft = state["captured_prebuffer2"]
    assert isinstance(captured_pre_mfcc, list)
    assert isinstance(captured_pre_fft, list)

    evento = np.array(captured_pre_mfcc + future_mfcc, dtype=np.float32)
    fft = np.array(captured_pre_fft + future_fft, dtype=np.float32)
    state["last_event_time"] = now
    state["recording"] = False
    return evento, fft


def _keyboard_record_thread(state: dict[str, object], buffers: dict[str, object], lock: threading.Lock) -> None:
    print("ENTER -> gravar referencia com pre-buffer de 5 s", flush=True)
    print("Ctrl+C -> sair", flush=True)
    while True:
        try:
            input()
        except EOFError:
            return

        now = time.time()
        with lock:
            armed = arm_recording(state, now, buffers)

        if armed:
            print("Gravando referencia de 10 s (5 s anteriores + 5 s posteriores)...", flush=True)
        else:
            print("Gravacao ja esta em andamento.", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Captura o I2S bruto espelhado pela FPGA, calcula FFT e MFCC no Raspberry "
            "e mantem o mesmo fluxo de referencia/comparacao."
        )
    )
    parser.add_argument("-D", "--device", default=DEFAULT_AUDIO_DEVICE, help="ALSA device, ex.: hw:2,0")
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=DEFAULT_CAPTURE_RATE_HZ,
        help="Taxa de amostragem nominal usada no host",
    )
    parser.add_argument(
        "--frame-bins",
        type=int,
        default=512,
        help="Numero de amostras por janela FFT calculada em software",
    )
    parser.add_argument(
        "--useful-bins",
        type=int,
        default=256,
        help="Quantidade de bins de magnitude mantidos para o comparador",
    )
    parser.add_argument(
        "--capture-backend",
        default=DEFAULT_CAPTURE_BACKEND,
        help="Backend de captura: auto, arecord ou alsa-c",
    )
    parser.add_argument(
        "--capture-binary",
        default=None,
        help="Caminho explicito para o helper nativo de captura ALSA",
    )
    parser.add_argument(
        "--channel-mode",
        default="auto",
        choices=("auto", "left", "right", "average"),
        help="Canal do stream I2S usado para a FFT em software",
    )
    parser.add_argument(
        "--sample-shift-bits",
        type=int,
        default=0,
        help="Shift aritmetico a direita aplicado nas amostras S32_LE antes da FFT",
    )
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Nao remove a componente DC da janela antes da FFT",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=RECORD_SECONDS,
        help="Duracao da parte posterior do snapshot de referencia",
    )
    parser.add_argument(
        "--prebuffer-seconds",
        type=float,
        default=PREBUFFER_SECONDS,
        help="Duracao do pre-buffer preservado antes do ENTER",
    )
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=HISTORY_SECONDS,
        help="Duracao do historico mantido para comparacao continua",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = FFTAdapterConfig(
        device=args.device,
        sample_rate=args.rate,
        frame_bins=args.frame_bins,
        useful_bins=args.useful_bins,
        capture_backend=args.capture_backend,
        capture_binary=args.capture_binary,
        channel_mode=args.channel_mode,
        sample_shift_bits=args.sample_shift_bits,
        remove_dc=not args.keep_dc,
    )

    rx = FPGAFFTReceiver(cfg)
    buffers = create_analysis_buffers(
        cfg.sample_rate,
        cfg.frame_bins,
        prebuffer_seconds=args.prebuffer_seconds,
        history_seconds=args.history_seconds,
    )
    state = create_runtime_state()
    lock = threading.Lock()

    threading.Thread(
        target=_keyboard_record_thread,
        args=(state, buffers, lock),
        daemon=True,
    ).start()
    threading.Thread(
        target=compararEvento,
        args=(buffers["history_mfcc"], buffers["history_fft"], lock, lambda: float(state["last_event_time"])),
        kwargs={"config": DirectComparatorConfig(useful_bins=cfg.useful_bins)},
        daemon=True,
    ).start()

    try:
        rx.start()
        print(
            f"Capturando audio bruto da FPGA em {cfg.sample_rate} Hz; "
            f"FFT software com janela de {cfg.frame_bins} amostras.",
            flush=True,
        )
        while True:
            frame = rx.read_frame()
            if frame is None:
                continue

            fft_bins, mfcc = frame
            now = time.time()

            with lock:
                snapshot = ingest_frame(
                    buffers,
                    state,
                    mfcc,
                    fft_bins,
                    now,
                    record_seconds=args.record_seconds,
                )

            if snapshot is None:
                continue

            evento, fft = snapshot
            save_event_snapshot(evento, fft)
            with lock:
                state["last_event_time"] = now
            print(
                f"Referencia salva: evento={evento.shape} fft={fft.shape} canal={rx.last_channel_used}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nEncerrando captura.", flush=True)
        return 0
    finally:
        rx.stop()


if __name__ == "__main__":
    raise SystemExit(main())
