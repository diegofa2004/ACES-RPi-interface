import argparse
import json
import math
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from .compararEvento import compararEvento
    from .fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
    from .i2s_stream import AUTO_AUDIO_DEVICE, build_arecord_cmd, resolve_audio_device
except ImportError:
    from compararEvento import compararEvento
    from fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
    from i2s_stream import AUTO_AUDIO_DEVICE, build_arecord_cmd, resolve_audio_device


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE") or AUTO_AUDIO_DEVICE
WORK_DIR = Path(__file__).resolve().parent
EVENTO_FILENAME = WORK_DIR / "evento.npy"
FFT_FILENAME = WORK_DIR / "fft.npy"
EVENTO_TMP_FILENAME = WORK_DIR / "evento_tmp.npy"
FFT_TMP_FILENAME = WORK_DIR / "fft_tmp.npy"

PREBUFFER_SECONDS = 5.0
HISTORY_SECONDS = 15.0
RECORD_SECONDS = 5.0
DEBUG_LOG_FORMAT_VERSION = 1
DEFAULT_DEBUG_CAPTURE_SECONDS = 10.0
DEFAULT_DEBUG_CHUNK_PAIRS = 1024
DEFAULT_DEBUG_PREVIEW_PAIRS = 12


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
) -> dict[str, deque]:
    buffer_size = frames_for_seconds(sample_rate, frame_bins, prebuffer_seconds)
    history_size = frames_for_seconds(sample_rate, frame_bins, history_seconds)
    return {
        "pre_mfcc": deque(maxlen=buffer_size),
        "history_mfcc": deque(maxlen=history_size),
        "pre_fft": deque(maxlen=buffer_size),
        "history_fft": deque(maxlen=history_size),
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


def arm_recording(state: dict[str, object], now: float, buffers: Optional[dict[str, deque]] = None) -> bool:
    if bool(state["recording"]):
        return False

    state["future_buffer"] = []
    state["future_buffer2"] = []
    state["captured_prebuffer"] = list(buffers["pre_mfcc"]) if buffers is not None else []
    state["captured_prebuffer2"] = list(buffers["pre_fft"]) if buffers is not None else []
    state["recording"] = True
    state["record_start"] = now
    return True


def ingest_frame(
    buffers: dict[str, deque],
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


def _u32_hex(word: int) -> str:
    return f"0x{int(word) & 0xFFFFFFFF:08X}"


def _decode_debug_word(word: int, cfg: FFTAdapterConfig) -> dict[str, object]:
    uword = int(word) & 0xFFFFFFFF
    tag = int((uword >> cfg.tag_shift) & cfg.tag_mask)

    payload_mask = (1 << cfg.payload_bits) - 1
    payload = int(uword & payload_mask)
    sign_bit = 1 << (cfg.payload_bits - 1)
    if payload & sign_bit:
        payload -= 1 << cfg.payload_bits

    reserved_width = max(0, cfg.tag_shift - cfg.payload_bits)
    reserved = 0
    if reserved_width > 0:
        reserved = int((uword >> cfg.payload_bits) & ((1 << reserved_width) - 1))

    return {
        "hex": _u32_hex(word),
        "i32": int(word),
        "tag": tag,
        "payload": payload,
        "reserved": reserved,
        "reserved_nonzero": bool(reserved),
    }


def _classify_debug_pair(left_tag: int, right_tag: int, cfg: FFTAdapterConfig) -> str:
    if left_tag != right_tag:
        return "tag_mismatch"
    if left_tag == cfg.tag_idle:
        return "idle"
    if left_tag == cfg.tag_bfpexp:
        return "bfpexp"
    if left_tag == cfg.tag_fft:
        return "fft"
    return "other"


def create_channel_debug_state() -> dict[str, object]:
    return {
        "chunk_index": 0,
        "total_pairs": 0,
        "kind_counts": Counter(),
        "raw_tag_counts_left": Counter(),
        "raw_tag_counts_right": Counter(),
        "transition_counts": Counter(),
        "max_run_by_kind": Counter(),
        "fft_run_lengths": [],
        "reserved_nonzero_words": 0,
        "flag_high_chunks": 0,
        "flag_low_chunks": 0,
        "flag_unknown_chunks": 0,
        "current_run_kind": None,
        "current_run_length": 0,
    }


def _finish_global_debug_run(state: dict[str, object]) -> None:
    current_kind = state["current_run_kind"]
    current_length = int(state["current_run_length"])
    if current_kind == "fft" and current_length > 0:
        fft_run_lengths = state["fft_run_lengths"]
        assert isinstance(fft_run_lengths, list)
        fft_run_lengths.append(current_length)


def finalize_channel_debug_state(state: dict[str, object]) -> None:
    _finish_global_debug_run(state)
    state["current_run_kind"] = None
    state["current_run_length"] = 0


def process_channel_debug_chunk(
    pairs: np.ndarray,
    cfg: FFTAdapterConfig,
    state: dict[str, object],
    *,
    preview_pairs: int,
    flag_active: Optional[bool],
    timestamp_ns: int,
) -> dict[str, object]:
    kind_counts = Counter()
    raw_tag_counts_left = Counter()
    raw_tag_counts_right = Counter()
    transition_counts = Counter()
    max_run_by_kind = Counter()
    fft_run_lengths = []
    preview = []
    reserved_nonzero_words = 0

    local_run_kind = None
    local_run_length = 0

    for pair_index, pair in enumerate(np.asarray(pairs, dtype=np.int32)):
        left = _decode_debug_word(int(pair[0]), cfg)
        right = _decode_debug_word(int(pair[1]), cfg)
        kind = _classify_debug_pair(int(left["tag"]), int(right["tag"]), cfg)

        kind_counts[kind] += 1
        raw_tag_counts_left[int(left["tag"])] += 1
        raw_tag_counts_right[int(right["tag"])] += 1
        reserved_nonzero_words += int(bool(left["reserved_nonzero"])) + int(bool(right["reserved_nonzero"]))

        if pair_index < preview_pairs:
            preview.append(
                {
                    "pair_index": pair_index,
                    "kind": kind,
                    "left": left,
                    "right": right,
                }
            )

        if local_run_kind != kind:
            if local_run_kind is not None:
                transition_counts[f"{local_run_kind}->{kind}"] += 1
                if local_run_kind == "fft":
                    fft_run_lengths.append(local_run_length)
            local_run_kind = kind
            local_run_length = 1
        else:
            local_run_length += 1
        max_run_by_kind[kind] = max(max_run_by_kind[kind], local_run_length)

        state_kind_counts = state["kind_counts"]
        state_raw_tag_counts_left = state["raw_tag_counts_left"]
        state_raw_tag_counts_right = state["raw_tag_counts_right"]
        state_transition_counts = state["transition_counts"]
        state_max_run_by_kind = state["max_run_by_kind"]
        assert isinstance(state_kind_counts, Counter)
        assert isinstance(state_raw_tag_counts_left, Counter)
        assert isinstance(state_raw_tag_counts_right, Counter)
        assert isinstance(state_transition_counts, Counter)
        assert isinstance(state_max_run_by_kind, Counter)

        state["total_pairs"] = int(state["total_pairs"]) + 1
        state_kind_counts[kind] += 1
        state_raw_tag_counts_left[int(left["tag"])] += 1
        state_raw_tag_counts_right[int(right["tag"])] += 1
        state["reserved_nonzero_words"] = int(state["reserved_nonzero_words"]) + int(bool(left["reserved_nonzero"]))
        state["reserved_nonzero_words"] = int(state["reserved_nonzero_words"]) + int(bool(right["reserved_nonzero"]))

        current_run_kind = state["current_run_kind"]
        current_run_length = int(state["current_run_length"])
        if current_run_kind != kind:
            if current_run_kind is not None:
                state_transition_counts[f"{current_run_kind}->{kind}"] += 1
                if current_run_kind == "fft" and current_run_length > 0:
                    fft_run_lengths_state = state["fft_run_lengths"]
                    assert isinstance(fft_run_lengths_state, list)
                    fft_run_lengths_state.append(current_run_length)
            state["current_run_kind"] = kind
            state["current_run_length"] = 1
        else:
            state["current_run_length"] = current_run_length + 1
        state_max_run_by_kind[kind] = max(state_max_run_by_kind[kind], int(state["current_run_length"]))

    if local_run_kind == "fft" and local_run_length > 0:
        fft_run_lengths.append(local_run_length)

    if flag_active is True:
        state["flag_high_chunks"] = int(state["flag_high_chunks"]) + 1
    elif flag_active is False:
        state["flag_low_chunks"] = int(state["flag_low_chunks"]) + 1
    else:
        state["flag_unknown_chunks"] = int(state["flag_unknown_chunks"]) + 1

    chunk_index = int(state["chunk_index"])
    state["chunk_index"] = chunk_index + 1

    return {
        "type": "chunk",
        "chunk_index": chunk_index,
        "timestamp_ns": int(timestamp_ns),
        "pair_count": int(len(pairs)),
        "flag_active": flag_active,
        "kind_counts": dict(sorted(kind_counts.items())),
        "raw_tag_counts_left": dict(sorted(raw_tag_counts_left.items())),
        "raw_tag_counts_right": dict(sorted(raw_tag_counts_right.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "max_run_by_kind": dict(sorted(max_run_by_kind.items())),
        "fft_run_lengths": fft_run_lengths,
        "reserved_nonzero_words": reserved_nonzero_words,
        "preview": preview,
    }


def build_channel_debug_summary(
    state: dict[str, object],
    *,
    duration_seconds: float,
    interrupted: bool,
    timestamp_ns: int,
) -> dict[str, object]:
    fft_run_lengths = list(state["fft_run_lengths"])
    fft_run_lengths.sort(reverse=True)

    return {
        "type": "summary",
        "timestamp_ns": int(timestamp_ns),
        "duration_seconds": float(duration_seconds),
        "interrupted": bool(interrupted),
        "chunk_count": int(state["chunk_index"]),
        "total_pairs": int(state["total_pairs"]),
        "kind_counts": dict(sorted(dict(state["kind_counts"]).items())),
        "raw_tag_counts_left": dict(sorted(dict(state["raw_tag_counts_left"]).items())),
        "raw_tag_counts_right": dict(sorted(dict(state["raw_tag_counts_right"]).items())),
        "transition_counts": dict(sorted(dict(state["transition_counts"]).items())),
        "max_run_by_kind": dict(sorted(dict(state["max_run_by_kind"]).items())),
        "top_fft_run_lengths": fft_run_lengths[:16],
        "fft_run_count": len(fft_run_lengths),
        "reserved_nonzero_words": int(state["reserved_nonzero_words"]),
        "flag_high_chunks": int(state["flag_high_chunks"]),
        "flag_low_chunks": int(state["flag_low_chunks"]),
        "flag_unknown_chunks": int(state["flag_unknown_chunks"]),
    }


def _write_jsonl_line(handle, payload: dict[str, object]) -> None:
    handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    handle.write("\n")
    handle.flush()


def run_channel_debug_capture(
    cfg: FFTAdapterConfig,
    *,
    device: str,
    log_path: Path,
    capture_seconds: float,
    chunk_pairs: int,
    preview_pairs: int,
) -> int:
    rx = FPGAFFTReceiver(cfg)
    try:
        rx.start()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

    log_path.parent.mkdir(parents=True, exist_ok=True)
    interrupted = False
    start_monotonic = time.monotonic()
    last_status = start_monotonic
    state = create_channel_debug_state()

    print("Channel debug mode active: passive capture, no frame protocol enforcement.", flush=True)
    print("Structured JSONL log:", log_path, flush=True)
    print("Commit this log file so we can inspect raw words, decoded tags, transitions, and FFT run lengths.", flush=True)

    try:
        with log_path.open("w", encoding="utf-8") as handle:
            _write_jsonl_line(
                handle,
                {
                    "type": "session_start",
                    "timestamp_ns": time.time_ns(),
                    "format_version": DEBUG_LOG_FORMAT_VERSION,
                    "mode": "passive_channel_debug",
                    "protocol_enforced": False,
                    "done_pulses_emitted": False,
                    "device": device,
                    "arecord_cmd": build_arecord_cmd(device, cfg.sample_rate),
                    "config": {
                        "sample_rate": cfg.sample_rate,
                        "frame_bins": cfg.frame_bins,
                        "useful_bins": cfg.useful_bins,
                        "use_i2s_tags": cfg.use_i2s_tags,
                        "tag_shift": cfg.tag_shift,
                        "tag_mask": cfg.tag_mask,
                        "payload_bits": cfg.payload_bits,
                        "tag_idle": cfg.tag_idle,
                        "tag_bfpexp": cfg.tag_bfpexp,
                        "tag_fft": cfg.tag_fft,
                        "require_bfpexp_before_fft": cfg.require_bfpexp_before_fft,
                        "bfpexp_flag_line": cfg.bfpexp_flag_line,
                        "done_line": cfg.done_line,
                    },
                    "capture_plan": {
                        "capture_seconds": capture_seconds,
                        "chunk_pairs": chunk_pairs,
                        "preview_pairs": preview_pairs,
                    },
                },
            )

            while True:
                elapsed = time.monotonic() - start_monotonic
                if elapsed >= capture_seconds:
                    break

                pairs = rx.read_available_pairs(chunk_pairs)
                if pairs is None:
                    if rx._proc is not None and rx._proc.poll() is not None:
                        break
                    continue

                flag_active = rx.read_flag_state()
                chunk_event = process_channel_debug_chunk(
                    pairs,
                    cfg,
                    state,
                    preview_pairs=preview_pairs,
                    flag_active=flag_active,
                    timestamp_ns=time.time_ns(),
                )
                _write_jsonl_line(handle, chunk_event)

                now = time.monotonic()
                if now - last_status >= 1.0:
                    kind_counts = chunk_event["kind_counts"]
                    assert isinstance(kind_counts, dict)
                    print(
                        "debug:",
                        f"chunks={state['chunk_index']}",
                        f"pairs={state['total_pairs']}",
                        f"idle={kind_counts.get('idle', 0)}",
                        f"bfpexp={kind_counts.get('bfpexp', 0)}",
                        f"fft={kind_counts.get('fft', 0)}",
                        f"mismatch={kind_counts.get('tag_mismatch', 0)}",
                        flush=True,
                    )
                    last_status = now

            finalize_channel_debug_state(state)
            _write_jsonl_line(
                handle,
                build_channel_debug_summary(
                    state,
                    duration_seconds=time.monotonic() - start_monotonic,
                    interrupted=interrupted,
                    timestamp_ns=time.time_ns(),
                ),
            )
    except KeyboardInterrupt:
        interrupted = True
        finalize_channel_debug_state(state)
        with log_path.open("a", encoding="utf-8") as handle:
            _write_jsonl_line(
                handle,
                build_channel_debug_summary(
                    state,
                    duration_seconds=time.monotonic() - start_monotonic,
                    interrupted=interrupted,
                    timestamp_ns=time.time_ns(),
                ),
            )
        print("Stopping debug capture...", flush=True)
    finally:
        rx.stop()

    print("Debug capture complete.", flush=True)
    return 0


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
    parser.add_argument(
        "--debug-channel-log",
        default=None,
        help="Write a passive JSONL channel debug log and exit after the capture window",
    )
    parser.add_argument(
        "--debug-capture-seconds",
        type=float,
        default=DEFAULT_DEBUG_CAPTURE_SECONDS,
        help="Duration of passive channel debug capture in seconds",
    )
    parser.add_argument(
        "--debug-chunk-pairs",
        type=int,
        default=DEFAULT_DEBUG_CHUNK_PAIRS,
        help="Number of raw stereo pairs summarized per JSONL chunk in debug mode",
    )
    parser.add_argument(
        "--debug-preview-pairs",
        type=int,
        default=DEFAULT_DEBUG_PREVIEW_PAIRS,
        help="Number of raw pairs previewed inside each JSONL debug chunk",
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
    if args.debug_capture_seconds <= 0:
        parser.error("--debug-capture-seconds must be positive")
    if args.debug_chunk_pairs <= 0:
        parser.error("--debug-chunk-pairs must be positive")
    if args.debug_preview_pairs < 0:
        parser.error("--debug-preview-pairs must be non-negative")

    try:
        device = resolve_audio_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    os.chdir(WORK_DIR)

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

    if args.debug_channel_log:
        return run_channel_debug_capture(
            cfg,
            device=device,
            log_path=Path(args.debug_channel_log),
            capture_seconds=args.debug_capture_seconds,
            chunk_pairs=args.debug_chunk_pairs,
            preview_pairs=args.debug_preview_pairs,
        )

    buffers = create_analysis_buffers(args.rate, args.frame_bins)
    lock = threading.Lock()
    state = create_runtime_state()

    def toggle_recording() -> None:
        while True:
            try:
                input()
            except EOFError:
                return

            with lock:
                armed = arm_recording(state, time.time(), buffers)
                if not armed:
                    continue

            print("Gravando evento com pre-buffer de 5 s...", flush=True)

    rx = FPGAFFTReceiver(cfg)
    try:
        rx.start()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

    threading.Thread(target=toggle_recording, daemon=True).start()
    threading.Thread(
        target=compararEvento,
        args=(buffers["history_mfcc"], buffers["history_fft"], lock, lambda: state["last_event_time"]),
        daemon=True,
    ).start()

    pre_size = buffers["pre_mfcc"].maxlen or 0
    history_size = buffers["history_mfcc"].maxlen or 0
    print("Using ALSA capture device:", device, flush=True)
    print("Reading FPGA FFT stream from I2S...", flush=True)
    if args.use_i2s_tags:
        print("Tagged mode: idle-tagged words are ignored while searching for frames.", flush=True)
        if args.allow_fft_without_bfpexp:
            print("Tagged mode sync: FFT tags may start a frame even without a BFPEXP tag.", flush=True)
        elif args.done_line is not None:
            print(
                "Tagged mode sync: startup can bootstrap from an in-flight FFT burst because DONE is configured.",
                flush=True,
            )
        else:
            print(
                "Tagged mode sync: waiting for BFPEXP before FFT frame start; "
                "if startup attaches mid-stream, use --done-line or --allow-fft-without-bfpexp.",
                flush=True,
            )
    print(
        "Buffer sizes:",
        f"pre_mfcc={pre_size}",
        f"history_mfcc={history_size}",
        f"pre_fft={buffers['pre_fft'].maxlen or 0}",
        f"history_fft={buffers['history_fft'].maxlen or 0}",
        flush=True,
    )
    print("Press ENTER to save an event like the pyserial flow.", flush=True)
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
