#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_DIO_SCK = 13
DEFAULT_DIO_WS = 14
DEFAULT_DIO_SD = 15
DEFAULT_SAMPLE_RATE_HZ = 100_000_000.0
DEFAULT_CAPTURE_SECONDS = 0.010
DEFAULT_BITS_PER_WORD = 32

ACQMODE_RECORD = ctypes.c_int(3)
TRIGSRC_NONE = ctypes.c_ubyte(0)


@dataclass(frozen=True)
class CaptureConfig:
    sample_rate_hz: float
    capture_seconds: float
    device_index: int
    dio_sck: int
    dio_ws: int
    dio_sd: int


@dataclass(frozen=True)
class DecodeConfig:
    sample_rate_hz: float
    dio_sck: int
    dio_ws: int
    dio_sd: int
    bits_per_word: int
    ws_low_channel: str
    sample_shift_bits: int = 0


@dataclass(frozen=True)
class DecodedWord:
    word_index: int
    sample_index: int
    time_s: float
    ws: int
    channel: str
    word: int
    value_signed: int
    sample_signed: int


@dataclass(frozen=True)
class StereoFrame:
    frame_index: int
    left: DecodedWord
    right: DecodedWord


@dataclass(frozen=True)
class DecodeSummary:
    words: list[DecodedWord]
    frames: list[StereoFrame]
    rising_edges: int
    ws_toggles: int
    error_counts: dict[str, int]


class DwfError(RuntimeError):
    pass


def _sign_extend(value: int, bits: int) -> int:
    if bits <= 0:
        return 0
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    value &= mask
    return (value ^ sign_bit) - sign_bit


def _channel_for_ws(ws: int, ws_low_channel: str) -> str:
    low = ws_low_channel.lower()
    if low not in {"left", "right"}:
        raise ValueError(f"ws_low_channel deve ser 'left' ou 'right', recebeu {ws_low_channel!r}")
    if ws == 0:
        return low
    return "right" if low == "left" else "left"


def decode_raw_word(word: int, *, bits_per_word: int, sample_shift_bits: int = 0) -> tuple[int, int]:
    value_signed = _sign_extend(int(word), bits_per_word)
    sample_signed = value_signed >> int(sample_shift_bits)
    return value_signed, sample_signed


def decode_i2s_words(samples: Sequence[int], config: DecodeConfig) -> DecodeSummary:
    prev_sck = None
    prev_edge_ws = 0
    prev_edge_ws_valid = False
    pending_start = False
    pending_ws = 0
    slot_ws = 0
    slot_shift = 0
    slot_count = 0
    word_mask = (1 << config.bits_per_word) - 1 if config.bits_per_word < 64 else None

    error_counts: Counter[str] = Counter()
    words: list[DecodedWord] = []
    rising_edges = 0
    ws_toggles = 0

    for sample_index, bus in enumerate(samples):
        sck = (int(bus) >> config.dio_sck) & 0x1
        ws = (int(bus) >> config.dio_ws) & 0x1
        sd = (int(bus) >> config.dio_sd) & 0x1

        if prev_sck is None:
            prev_sck = sck
            continue

        if prev_sck == 0 and sck == 1:
            rising_edges += 1
            ws_changed = prev_edge_ws_valid and ws != prev_edge_ws
            if ws_changed:
                ws_toggles += 1

            if pending_start:
                if ws_changed:
                    error_counts["ws_changed_again_before_word_start"] += 1
                if ws != pending_ws:
                    error_counts["ws_unstable_before_word_start"] += 1

                slot_ws = pending_ws
                slot_shift = sd
                slot_count = 1
                pending_start = False
            elif slot_count != 0:
                completed_word = (slot_shift << 1) | sd
                if word_mask is not None:
                    completed_word &= word_mask

                if slot_count == config.bits_per_word - 1:
                    if not ws_changed:
                        error_counts["ws_missing_on_last_bit"] += 1
                    else:
                        value_signed, sample_signed = decode_raw_word(
                            completed_word,
                            bits_per_word=config.bits_per_word,
                            sample_shift_bits=config.sample_shift_bits,
                        )
                        words.append(
                            DecodedWord(
                                word_index=len(words),
                                sample_index=sample_index,
                                time_s=sample_index / config.sample_rate_hz,
                                ws=slot_ws,
                                channel=_channel_for_ws(slot_ws, config.ws_low_channel),
                                word=completed_word,
                                value_signed=value_signed,
                                sample_signed=sample_signed,
                            )
                        )
                    slot_count = 0
                    slot_shift = 0
                else:
                    if ws_changed:
                        error_counts["ws_changed_early"] += 1
                    slot_shift = completed_word
                    slot_count += 1

            if ws_changed:
                if pending_start:
                    error_counts["ws_changed_while_start_pending"] += 1
                pending_start = True
                pending_ws = ws

            prev_edge_ws_valid = True
            prev_edge_ws = ws

        prev_sck = sck

    frames = pair_stereo_words(words)
    return DecodeSummary(
        words=words,
        frames=frames,
        rising_edges=rising_edges,
        ws_toggles=ws_toggles,
        error_counts=dict(error_counts),
    )


def pair_stereo_words(words: Sequence[DecodedWord]) -> list[StereoFrame]:
    frames: list[StereoFrame] = []
    pending: DecodedWord | None = None

    for word in words:
        if pending is None:
            pending = word
            continue

        if pending.channel == word.channel:
            pending = word
            continue

        left = pending if pending.channel == "left" else word
        right = pending if pending.channel == "right" else word
        frames.append(StereoFrame(frame_index=len(frames), left=left, right=right))
        pending = None

    return frames


def _open_dwf() -> ctypes.CDLL:
    if sys.platform.startswith("win"):
        for candidate in (
            "dwf",
            r"C:\Program Files (x86)\Digilent\WaveForms3\dwf.dll",
            r"C:\Program Files\Digilent\WaveForms3\dwf.dll",
        ):
            try:
                return ctypes.cdll.LoadLibrary(candidate)
            except OSError:
                continue
        raise DwfError("Nao foi possivel carregar o dwf.dll. Instale o WaveForms SDK ou ajuste o PATH.")

    if sys.platform.startswith("darwin"):
        return ctypes.cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")

    return ctypes.cdll.LoadLibrary("libdwf.so")


def _dwf_ok(dwf: ctypes.CDLL, status: int, context: str) -> None:
    if status == 1:
        return
    message = ctypes.create_string_buffer(512)
    try:
        dwf.FDwfGetLastErrorMsg(message)
        detail = message.value.decode(errors="replace")
    except Exception:
        detail = "sem mensagem de erro da biblioteca"
    raise DwfError(f"{context} falhou: {detail}")


def capture_samples_with_waveforms(config: CaptureConfig) -> tuple[list[int], float, dict[str, int | float]]:
    dwf = _open_dwf()
    hdwf = ctypes.c_int()
    version = ctypes.create_string_buffer(32)
    dwf.FDwfGetVersion(version)
    _dwf_ok(dwf, dwf.FDwfDeviceOpen(ctypes.c_int(config.device_index), ctypes.byref(hdwf)), "FDwfDeviceOpen")

    if hdwf.value == 0:
        raise DwfError("Nenhum Analog Discovery foi aberto.")

    try:
        _dwf_ok(dwf, dwf.FDwfDeviceAutoConfigureSet(hdwf, ctypes.c_int(0)), "FDwfDeviceAutoConfigureSet")

        base_hz = ctypes.c_double()
        _dwf_ok(dwf, dwf.FDwfDigitalInInternalClockInfo(hdwf, ctypes.byref(base_hz)), "FDwfDigitalInInternalClockInfo")
        divider = max(1, int(round(base_hz.value / config.sample_rate_hz)))

        _dwf_ok(dwf, dwf.FDwfDigitalInAcquisitionModeSet(hdwf, ACQMODE_RECORD), "FDwfDigitalInAcquisitionModeSet")
        _dwf_ok(dwf, dwf.FDwfDigitalInDividerSet(hdwf, ctypes.c_int(divider)), "FDwfDigitalInDividerSet")
        _dwf_ok(dwf, dwf.FDwfDigitalInSampleFormatSet(hdwf, ctypes.c_int(16)), "FDwfDigitalInSampleFormatSet")
        _dwf_ok(dwf, dwf.FDwfDigitalInTriggerSourceSet(hdwf, TRIGSRC_NONE), "FDwfDigitalInTriggerSourceSet")
        _dwf_ok(dwf, dwf.FDwfDigitalInTriggerPositionSet(hdwf, ctypes.c_int(0)), "FDwfDigitalInTriggerPositionSet")
        _dwf_ok(dwf, dwf.FDwfDigitalInConfigure(hdwf, ctypes.c_int(0), ctypes.c_int(1)), "FDwfDigitalInConfigure(start)")

        actual_divider = ctypes.c_int()
        _dwf_ok(dwf, dwf.FDwfDigitalInDividerGet(hdwf, ctypes.byref(actual_divider)), "FDwfDigitalInDividerGet")
        actual_sample_rate_hz = base_hz.value / max(actual_divider.value, 1)
        target_samples = max(1, int(round(actual_sample_rate_hz * config.capture_seconds)))

        captured: list[int] = []
        lost_total = 0
        corrupted_total = 0
        started = time.monotonic()

        while len(captured) < target_samples:
            status = ctypes.c_ubyte()
            available = ctypes.c_int()
            lost = ctypes.c_int()
            corrupted = ctypes.c_int()

            _dwf_ok(dwf, dwf.FDwfDigitalInStatus(hdwf, ctypes.c_int(1), ctypes.byref(status)), "FDwfDigitalInStatus")
            _dwf_ok(
                dwf,
                dwf.FDwfDigitalInStatusRecord(
                    hdwf,
                    ctypes.byref(available),
                    ctypes.byref(lost),
                    ctypes.byref(corrupted),
                ),
                "FDwfDigitalInStatusRecord",
            )

            lost_total += lost.value
            corrupted_total += corrupted.value

            if available.value <= 0:
                if time.monotonic() - started > max(config.capture_seconds * 4.0, 1.0):
                    break
                time.sleep(0.001)
                continue

            chunk = min(available.value, target_samples - len(captured))
            buffer = (ctypes.c_uint16 * chunk)()
            _dwf_ok(dwf, dwf.FDwfDigitalInStatusData(hdwf, ctypes.byref(buffer), ctypes.c_int(2 * chunk)), "FDwfDigitalInStatusData")
            captured.extend(int(value) for value in buffer)

        _dwf_ok(dwf, dwf.FDwfDigitalInConfigure(hdwf, ctypes.c_int(0), ctypes.c_int(0)), "FDwfDigitalInConfigure(stop)")

        metadata: dict[str, int | float] = {
            "requested_sample_rate_hz": config.sample_rate_hz,
            "actual_sample_rate_hz": actual_sample_rate_hz,
            "base_clock_hz": base_hz.value,
            "divider": actual_divider.value,
            "captured_samples": len(captured),
            "lost_samples": lost_total,
            "corrupted_samples": corrupted_total,
        }
        return captured, actual_sample_rate_hz, metadata
    finally:
        dwf.FDwfDeviceClose(hdwf)


def write_samples_csv(
    path: Path,
    samples: Sequence[int],
    decode_summary: DecodeSummary,
    config: DecodeConfig,
) -> None:
    word_by_sample = {word.sample_index: word for word in decode_summary.words}
    prev_sck = None
    sck_name = f"dio{config.dio_sck}_sck"
    ws_name = f"dio{config.dio_ws}_ws"
    sd_name = f"dio{config.dio_sd}_sd"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "time_s",
                sck_name,
                ws_name,
                sd_name,
                "bus_hex",
                "sck_rising",
                "decoded_word_hex",
                "decoded_channel",
                "decoded_value_signed",
                "decoded_sample_signed",
            ]
        )

        for sample_index, bus in enumerate(samples):
            sck = (int(bus) >> config.dio_sck) & 0x1
            ws = (int(bus) >> config.dio_ws) & 0x1
            sd = (int(bus) >> config.dio_sd) & 0x1
            rising = 1 if prev_sck == 0 and sck == 1 else 0
            decoded = word_by_sample.get(sample_index)

            writer.writerow(
                [
                    sample_index,
                    f"{sample_index / config.sample_rate_hz:.12f}",
                    sck,
                    ws,
                    sd,
                    f"0x{int(bus) & 0xFFFF:04X}",
                    rising,
                    "" if decoded is None else f"0x{decoded.word & 0xFFFFFFFF:08X}",
                    "" if decoded is None else decoded.channel,
                    "" if decoded is None else decoded.value_signed,
                    "" if decoded is None else decoded.sample_signed,
                ]
            )
            prev_sck = sck


def write_words_csv(path: Path, words: Sequence[DecodedWord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "word_index",
                "sample_index",
                "time_s",
                "ws",
                "channel",
                "word_hex",
                "value_signed",
                "sample_signed",
            ]
        )
        for word in words:
            writer.writerow(
                [
                    word.word_index,
                    word.sample_index,
                    f"{word.time_s:.12f}",
                    word.ws,
                    word.channel,
                    f"0x{word.word & 0xFFFFFFFF:08X}",
                    word.value_signed,
                    word.sample_signed,
                ]
            )


def write_frames_csv(path: Path, frames: Sequence[StereoFrame]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "left_word_hex",
                "left_value_signed",
                "left_sample_signed",
                "left_sample_index",
                "right_word_hex",
                "right_value_signed",
                "right_sample_signed",
                "right_sample_index",
            ]
        )
        for frame in frames:
            writer.writerow(
                [
                    frame.frame_index,
                    f"0x{frame.left.word & 0xFFFFFFFF:08X}",
                    frame.left.value_signed,
                    frame.left.sample_signed,
                    frame.left.sample_index,
                    f"0x{frame.right.word & 0xFFFFFFFF:08X}",
                    frame.right.value_signed,
                    frame.right.sample_signed,
                    frame.right.sample_index,
                ]
            )


def write_frames_raw(path: Path, frames: Sequence[StereoFrame]) -> None:
    if not frames:
        path.write_bytes(b"")
        return
    stereo = np.asarray(
        [(frame.left.sample_signed, frame.right.sample_signed) for frame in frames],
        dtype=np.int32,
    )
    path.write_bytes(stereo.tobytes())


def _default_output_prefix() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("pi_logs") / "analog" / f"ad_i2s_capture_{timestamp}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the raw mirrored microphone I2S bus with Analog Discovery and export decoded stereo files."
    )
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ, help="Digital capture sample rate")
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=DEFAULT_CAPTURE_SECONDS,
        help="Requested capture duration in seconds",
    )
    parser.add_argument("--device-index", type=int, default=-1, help="WaveForms device index, -1 opens the first one")
    parser.add_argument("--dio-sck", type=int, default=DEFAULT_DIO_SCK, help="DIO index used by I2S clock")
    parser.add_argument("--dio-ws", type=int, default=DEFAULT_DIO_WS, help="DIO index used by I2S word select")
    parser.add_argument("--dio-sd", type=int, default=DEFAULT_DIO_SD, help="DIO index used by I2S serial data")
    parser.add_argument("--bits-per-word", type=int, default=DEFAULT_BITS_PER_WORD, help="I2S slot width in bits")
    parser.add_argument(
        "--ws-low-channel",
        choices=("left", "right"),
        default="left",
        help="Channel represented when WS is low",
    )
    parser.add_argument(
        "--sample-shift-bits",
        type=int,
        default=0,
        help="Arithmetic right shift applied to each decoded word before exporting the stereo raw file",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix without extension. Defaults to pi_logs/analog/ad_i2s_capture_<timestamp>",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.sample_rate_hz <= 0:
        raise SystemExit("--sample-rate-hz deve ser positivo")
    if args.capture_seconds <= 0:
        raise SystemExit("--capture-seconds deve ser positivo")
    if not 1 <= args.bits_per_word <= 32:
        raise SystemExit("--bits-per-word deve ficar entre 1 e 32")
    if args.sample_shift_bits < 0:
        raise SystemExit("--sample-shift-bits deve ser nao negativo")

    output_prefix = args.output_prefix or _default_output_prefix()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    capture_config = CaptureConfig(
        sample_rate_hz=args.sample_rate_hz,
        capture_seconds=args.capture_seconds,
        device_index=args.device_index,
        dio_sck=args.dio_sck,
        dio_ws=args.dio_ws,
        dio_sd=args.dio_sd,
    )

    try:
        samples, actual_sample_rate_hz, capture_meta = capture_samples_with_waveforms(capture_config)
    except DwfError as exc:
        print(f"Erro no WaveForms: {exc}", file=sys.stderr)
        return 1

    decode_config = DecodeConfig(
        sample_rate_hz=actual_sample_rate_hz,
        dio_sck=args.dio_sck,
        dio_ws=args.dio_ws,
        dio_sd=args.dio_sd,
        bits_per_word=args.bits_per_word,
        ws_low_channel=args.ws_low_channel,
        sample_shift_bits=args.sample_shift_bits,
    )
    summary = decode_i2s_words(samples, decode_config)

    samples_csv = output_prefix.with_name(output_prefix.name + "_samples.csv")
    words_csv = output_prefix.with_name(output_prefix.name + "_words.csv")
    frames_csv = output_prefix.with_name(output_prefix.name + "_frames.csv")
    frames_raw = output_prefix.with_name(output_prefix.name + "_frames.raw")
    summary_txt = output_prefix.with_name(output_prefix.name + "_summary.txt")

    write_samples_csv(samples_csv, samples, summary, decode_config)
    write_words_csv(words_csv, summary.words)
    write_frames_csv(frames_csv, summary.frames)
    write_frames_raw(frames_raw, summary.frames)

    left_mean_abs = (
        float(np.mean(np.abs([frame.left.sample_signed for frame in summary.frames]), dtype=np.float64))
        if summary.frames
        else 0.0
    )
    right_mean_abs = (
        float(np.mean(np.abs([frame.right.sample_signed for frame in summary.frames]), dtype=np.float64))
        if summary.frames
        else 0.0
    )

    with summary_txt.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"requested_sample_rate_hz={args.sample_rate_hz}\n")
        handle.write(f"actual_sample_rate_hz={actual_sample_rate_hz}\n")
        handle.write(f"capture_seconds={args.capture_seconds}\n")
        handle.write(f"sample_shift_bits={args.sample_shift_bits}\n")
        handle.write(f"captured_samples={capture_meta['captured_samples']}\n")
        handle.write(f"lost_samples={capture_meta['lost_samples']}\n")
        handle.write(f"corrupted_samples={capture_meta['corrupted_samples']}\n")
        handle.write(f"rising_edges={summary.rising_edges}\n")
        handle.write(f"ws_toggles={summary.ws_toggles}\n")
        handle.write(f"decoded_words={len(summary.words)}\n")
        handle.write(f"decoded_frames={len(summary.frames)}\n")
        handle.write(f"left_mean_abs={left_mean_abs}\n")
        handle.write(f"right_mean_abs={right_mean_abs}\n")
        for key, value in sorted(summary.error_counts.items()):
            handle.write(f"error_{key}={value}\n")

    print(f"Captura salva em: {samples_csv}")
    print(f"Palavras decodificadas em: {words_csv}")
    print(f"Frames estereo em: {frames_csv}")
    print(f"Raw estereo em: {frames_raw}")
    print(f"Resumo em: {summary_txt}")
    print(
        "Amostras="
        f"{capture_meta['captured_samples']} "
        f"sample_rate={actual_sample_rate_hz:.3f}Hz "
        f"words={len(summary.words)} "
        f"frames={len(summary.frames)} "
        f"lost={capture_meta['lost_samples']} "
        f"corrupted={capture_meta['corrupted_samples']}"
    )
    if summary.error_counts:
        print("Erros de protocolo detectados:")
        for key, value in sorted(summary.error_counts.items()):
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
