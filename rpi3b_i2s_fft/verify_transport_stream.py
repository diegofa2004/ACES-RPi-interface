#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, TextIO

import numpy as np

try:
    from .i2s_stream import (
        DEFAULT_FFT_PACKET_INDEX_BASE,
        DEFAULT_PACKET_INDEX_BITS,
        DEFAULT_PACKET_INDEX_SHIFT,
        DEFAULT_PAYLOAD_BITS,
        DEFAULT_TAG_BFPEXP,
        DEFAULT_TAG_FFT,
        DEFAULT_TAG_IDLE,
        DEFAULT_TAG_MASK,
        DEFAULT_TAG_SHIFT,
        classify_tagged_i2s_pair,
        decode_tagged_i2s_word,
    )
except ImportError:
    from i2s_stream import (
        DEFAULT_FFT_PACKET_INDEX_BASE,
        DEFAULT_PACKET_INDEX_BITS,
        DEFAULT_PACKET_INDEX_SHIFT,
        DEFAULT_PAYLOAD_BITS,
        DEFAULT_TAG_BFPEXP,
        DEFAULT_TAG_FFT,
        DEFAULT_TAG_IDLE,
        DEFAULT_TAG_MASK,
        DEFAULT_TAG_SHIFT,
        classify_tagged_i2s_pair,
        decode_tagged_i2s_word,
    )


DEFAULT_CAPTURE_BINARY = str(Path("~/Desktop/alsa_log").expanduser())
DEFAULT_CAPTURE_DEVICE = "hw:2,0"
DEFAULT_CAPTURE_RATE_HZ = 48828
DEFAULT_CAPTURE_READ_FRAMES = 512


@dataclass(frozen=True)
class TaggedTransportConfig:
    frame_bins: int = 512
    useful_bins: int = 256
    packet_index_shift: int = DEFAULT_PACKET_INDEX_SHIFT
    packet_index_bits: int = DEFAULT_PACKET_INDEX_BITS
    fft_packet_index_base: int = DEFAULT_FFT_PACKET_INDEX_BASE
    tag_shift: int = DEFAULT_TAG_SHIFT
    tag_mask: int = DEFAULT_TAG_MASK
    payload_bits: int = DEFAULT_PAYLOAD_BITS
    tag_idle: int = DEFAULT_TAG_IDLE
    tag_bfpexp: int = DEFAULT_TAG_BFPEXP
    tag_fft: int = DEFAULT_TAG_FFT
    bfpexp_pairs_required: int = 128
    allow_fft_without_bfpexp: bool = False
    loss_tolerance_pairs: int = 0


@dataclass(frozen=True)
class DecodedWord:
    hex_value: str
    tag: int
    packet_index: int
    payload: int
    reserved: int
    reserved_nonzero: bool


@dataclass(frozen=True)
class FrameReport:
    frame_index: int
    bfpexp_value: int
    bfpexp_run_length: int
    fft_run_length: int
    peak_bin: int
    peak_magnitude: float
    shift_from_prev: Optional[int]


@dataclass(frozen=True)
class TransportSummary:
    pair_count: int
    kind_counts: dict[str, int]
    transition_counts: dict[str, int]
    run_hist_by_kind: dict[str, dict[int, int]]
    mismatch_count: int
    packet_index_mismatch_count: int
    unknown_count: int
    reserved_nonzero_words: int
    bfpexp_to_fft_examples: tuple[tuple[int, int], ...]
    frame_reports: tuple[FrameReport, ...]


def _u32_hex(word: int) -> str:
    return f"0x{int(word) & 0xFFFFFFFF:08X}"


def decode_tagged_word(word: int, cfg: TaggedTransportConfig) -> DecodedWord:
    decoded = decode_tagged_i2s_word(
        word,
        packet_index_shift=cfg.packet_index_shift,
        packet_index_bits=cfg.packet_index_bits,
        tag_shift=cfg.tag_shift,
        tag_mask=cfg.tag_mask,
        payload_bits=cfg.payload_bits,
    )

    return DecodedWord(
        hex_value=_u32_hex(word),
        tag=int(decoded["tag"]),
        packet_index=int(decoded["packet_index"]),
        payload=int(decoded["payload"]),
        reserved=int(decoded["reserved"]),
        reserved_nonzero=bool(decoded["reserved_nonzero"]),
    )


def classify_tagged_pair(
    left_tag: int,
    right_tag: int,
    cfg: TaggedTransportConfig,
    *,
    left_packet_index: int,
    right_packet_index: int,
) -> str:
    return classify_tagged_i2s_pair(
        left_tag,
        right_tag,
        left_packet_index=left_packet_index,
        right_packet_index=right_packet_index,
        tag_idle=cfg.tag_idle,
        tag_bfpexp=cfg.tag_bfpexp,
        tag_fft=cfg.tag_fft,
        fft_packet_index_base=cfg.fft_packet_index_base,
    )


def classify_tagged_pair_words(left: DecodedWord, right: DecodedWord, cfg: TaggedTransportConfig) -> str:
    return classify_tagged_pair(
        left.tag,
        right.tag,
        cfg,
        left_packet_index=left.packet_index,
        right_packet_index=right.packet_index,
    )


def is_tolerable_loss_kind(kind: str) -> bool:
    return kind in ("tag_mismatch", "packet_index_mismatch", "unknown_tag", "idle")


def create_contract_tracker(cfg: TaggedTransportConfig) -> dict[str, object]:
    return {
        "frame_bins": int(cfg.frame_bins),
        "bfpexp_hold_pairs": int(cfg.bfpexp_pairs_required),
        "allow_fft_without_bfpexp": bool(cfg.allow_fft_without_bfpexp),
        "loss_tolerance_pairs": int(cfg.loss_tolerance_pairs),
        "frame_number": 0,
        "bfpexp_run": 0,
        "bfpexp_loss_run": 0,
        "fft_index": 0,
        "fft_loss_run": 0,
        "inside_fft": False,
        "bootstrapped": False,
    }


def advance_contract_tracker(
    tracker: dict[str, object],
    kind: str,
) -> tuple[str, int, int]:
    frame_number = int(tracker["frame_number"])

    if bool(tracker["inside_fft"]):
        if kind == "fft":
            fft_index = int(tracker["fft_index"])
            tracker["fft_index"] = fft_index + 1
            phase = "fft_frame_bootstrap" if bool(tracker["bootstrapped"]) else "fft_frame"
            if int(tracker["fft_index"]) >= int(tracker["frame_bins"]):
                tracker["inside_fft"] = False
                tracker["fft_index"] = 0
                tracker["bfpexp_run"] = 0
                tracker["bfpexp_loss_run"] = 0
                tracker["fft_loss_run"] = 0
                tracker["bootstrapped"] = False
                tracker["frame_number"] = frame_number + 1
            return phase, frame_number, fft_index
        if is_tolerable_loss_kind(kind) and int(tracker["fft_loss_run"]) < int(tracker["loss_tolerance_pairs"]):
            fft_index = int(tracker["fft_index"])
            tracker["fft_index"] = fft_index + 1
            tracker["fft_loss_run"] = int(tracker["fft_loss_run"]) + 1
            phase = "fft_frame_gap_bootstrap" if bool(tracker["bootstrapped"]) else "fft_frame_gap"
            if int(tracker["fft_index"]) >= int(tracker["frame_bins"]):
                tracker["inside_fft"] = False
                tracker["fft_index"] = 0
                tracker["bfpexp_run"] = 0
                tracker["bfpexp_loss_run"] = 0
                tracker["fft_loss_run"] = 0
                tracker["bootstrapped"] = False
                tracker["frame_number"] = frame_number + 1
            return phase, frame_number, fft_index

        tracker["inside_fft"] = False
        tracker["fft_index"] = 0
        tracker["fft_loss_run"] = 0
        tracker["bootstrapped"] = False
        if kind == "bfpexp":
            tracker["bfpexp_run"] = 1
            tracker["bfpexp_loss_run"] = 0
            return "protocol_reset_bfpexp", frame_number, 0
        tracker["bfpexp_run"] = 0
        tracker["bfpexp_loss_run"] = 0
        return f"protocol_reset_{kind}", frame_number, -1

    if kind == "idle":
        if int(tracker["bfpexp_run"]) == 0:
            return "search_idle", frame_number, -1
        if int(tracker["bfpexp_loss_run"]) < int(tracker["loss_tolerance_pairs"]):
            bfpexp_index = int(tracker["bfpexp_run"])
            tracker["bfpexp_run"] = bfpexp_index + 1
            tracker["bfpexp_loss_run"] = int(tracker["bfpexp_loss_run"]) + 1
            return "bfpexp_preamble_gap", frame_number, bfpexp_index
        if int(tracker["bfpexp_run"]) != 0:
            tracker["bfpexp_run"] = 0
            tracker["bfpexp_loss_run"] = 0
        return "search_idle", frame_number, -1

    if kind == "bfpexp":
        bfpexp_index = int(tracker["bfpexp_run"])
        tracker["bfpexp_run"] = bfpexp_index + 1
        tracker["bfpexp_loss_run"] = 0
        return "bfpexp_preamble", frame_number, bfpexp_index

    if is_tolerable_loss_kind(kind):
        if int(tracker["bfpexp_run"]) > 0 and int(tracker["bfpexp_loss_run"]) < int(tracker["loss_tolerance_pairs"]):
            bfpexp_index = int(tracker["bfpexp_run"])
            tracker["bfpexp_run"] = bfpexp_index + 1
            tracker["bfpexp_loss_run"] = int(tracker["bfpexp_loss_run"]) + 1
            return "bfpexp_preamble_gap", frame_number, bfpexp_index
        tracker["bfpexp_run"] = 0
        tracker["bfpexp_loss_run"] = 0
        return f"protocol_reset_{kind}", frame_number, -1

    if kind == "fft":
        bfpexp_run = int(tracker["bfpexp_run"])
        bootstrap = bfpexp_run == 0 and bool(tracker["allow_fft_without_bfpexp"])
        if (not bootstrap) and (bfpexp_run < int(tracker["bfpexp_hold_pairs"])):
            tracker["bfpexp_run"] = 0
            tracker["bfpexp_loss_run"] = 0
            return "protocol_wait_bfpexp", frame_number, -1

        tracker["inside_fft"] = True
        tracker["bootstrapped"] = bootstrap
        tracker["fft_index"] = 1
        tracker["fft_loss_run"] = 0
        phase = "fft_frame_bootstrap" if bootstrap else "fft_frame"
        if int(tracker["frame_bins"]) == 1:
            tracker["inside_fft"] = False
            tracker["fft_index"] = 0
            tracker["bfpexp_run"] = 0
            tracker["bfpexp_loss_run"] = 0
            tracker["fft_loss_run"] = 0
            tracker["bootstrapped"] = False
            tracker["frame_number"] = frame_number + 1
        return phase, frame_number, 0

    return "protocol_unknown", frame_number, -1


def parse_hex_pair_line(line: str, cfg: TaggedTransportConfig) -> Optional[tuple[DecodedWord, DecodedWord, str]]:
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) != 2:
        return None
    left = decode_tagged_word(int(parts[0], 16), cfg)
    right = decode_tagged_word(int(parts[1], 16), cfg)
    kind = classify_tagged_pair_words(left, right, cfg)
    return left, right, kind


def _best_circular_shift(previous_pairs: Sequence[tuple[int, int]], current_pairs: Sequence[tuple[int, int]], useful_bins: int) -> int:
    usable = min(len(previous_pairs), len(current_pairs), useful_bins)
    prev = np.asarray(previous_pairs[:usable], dtype=np.float32)
    curr = np.asarray(current_pairs[:usable], dtype=np.float32)
    prev_mag = np.sqrt(prev[:, 0] * prev[:, 0] + prev[:, 1] * prev[:, 1])
    curr_mag = np.sqrt(curr[:, 0] * curr[:, 0] + curr[:, 1] * curr[:, 1])
    prev_mag -= float(np.mean(prev_mag))
    curr_mag -= float(np.mean(curr_mag))
    scores = [
        float(np.dot(prev_mag, np.roll(curr_mag, -shift)))
        for shift in range(usable)
    ]
    shift = int(np.argmax(scores))
    if shift > (usable // 2):
        shift -= usable
    return shift


def _peak_from_pairs(pairs: Sequence[tuple[int, int]], useful_bins: int) -> tuple[int, float]:
    usable = min(len(pairs), useful_bins)
    data = np.asarray(pairs[:usable], dtype=np.float32)
    magnitude = np.sqrt(data[:, 0] * data[:, 0] + data[:, 1] * data[:, 1])
    peak_bin = int(np.argmax(magnitude))
    return peak_bin, float(magnitude[peak_bin])


class TransportMonitor:
    def __init__(
        self,
        cfg: TaggedTransportConfig,
        *,
        print_limit: int,
        printer: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = cfg
        self._print_limit = int(print_limit)
        self._printer = printer or print
        self._printed_pairs = 0
        self._tracker = create_contract_tracker(cfg)

        self.pair_count = 0
        self.kind_counts: Counter[str] = Counter()
        self.transition_counts: Counter[str] = Counter()
        self.run_hist_by_kind: dict[str, Counter[int]] = {
            "idle": Counter(),
            "bfpexp": Counter(),
            "fft": Counter(),
            "tag_mismatch": Counter(),
            "packet_index_mismatch": Counter(),
            "unknown_tag": Counter(),
        }
        self.mismatch_count = 0
        self.packet_index_mismatch_count = 0
        self.unknown_count = 0
        self.reserved_nonzero_words = 0
        self.bfpexp_to_fft_examples: list[tuple[int, int]] = []
        self.frame_reports: list[FrameReport] = []

        self._current_run_kind: Optional[str] = None
        self._current_run_length = 0
        self._current_run_pairs: list[tuple[int, int]] = []

        self._previous_run_kind: Optional[str] = None
        self._previous_run_length = 0
        self._previous_run_pairs: list[tuple[int, int]] = []
        self._previous_full_frame_pairs: Optional[list[tuple[int, int]]] = None

    def _print_pair(
        self,
        left: DecodedWord,
        right: DecodedWord,
        kind: str,
        phase: str,
        frame_number: int,
        phase_index: int,
    ) -> None:
        if self._print_limit > 0 and self._printed_pairs >= self._print_limit:
            return
        line = (
            f"{self.pair_count:06d} "
            f"{left.hex_value} {right.hex_value} "
            f"kind={kind:<12} phase={phase:<24} "
            f"frame={frame_number:03d} idx={phase_index:03d} "
            f"pkt={left.packet_index:03d}/{right.packet_index:03d} "
            f"real={left.payload:8d} imag={right.payload:8d}"
        )
        if left.reserved_nonzero or right.reserved_nonzero:
            line += " reserved!=0"
        self._printer(line)
        self._printed_pairs += 1

    def _build_frame_report(
        self,
        frame_pairs: list[tuple[int, int]],
        *,
        bfpexp_value: int,
        bfpexp_run_length: int,
        fft_run_length: int,
    ) -> FrameReport:
        peak_bin, peak_magnitude = _peak_from_pairs(frame_pairs, self.cfg.useful_bins)
        shift_from_prev = None
        if self._previous_full_frame_pairs is not None:
            shift_from_prev = _best_circular_shift(
                self._previous_full_frame_pairs,
                frame_pairs,
                self.cfg.useful_bins,
            )
        report = FrameReport(
            frame_index=len(self.frame_reports),
            bfpexp_value=int(bfpexp_value),
            bfpexp_run_length=int(bfpexp_run_length),
            fft_run_length=int(fft_run_length),
            peak_bin=peak_bin,
            peak_magnitude=peak_magnitude,
            shift_from_prev=shift_from_prev,
        )
        self._previous_full_frame_pairs = list(frame_pairs)
        return report

    def _finish_current_run(self) -> None:
        if self._current_run_kind is None:
            return

        kind = self._current_run_kind
        length = int(self._current_run_length)
        run_pairs = list(self._current_run_pairs)
        self.run_hist_by_kind.setdefault(kind, Counter())[length] += 1

        if self._previous_run_kind == "bfpexp" and kind == "fft":
            self.bfpexp_to_fft_examples.append((self._previous_run_length, length))
            if self._previous_run_length >= self.cfg.bfpexp_pairs_required and length >= self.cfg.frame_bins:
                bfpexp_value = 0
                if self._previous_run_pairs:
                    bfpexp_value = int(self._previous_run_pairs[0][0])
                frame_pairs = run_pairs[: self.cfg.frame_bins]
                report = self._build_frame_report(
                    frame_pairs,
                    bfpexp_value=bfpexp_value,
                    bfpexp_run_length=self._previous_run_length,
                    fft_run_length=length,
                )
                self.frame_reports.append(report)

        self._previous_run_kind = kind
        self._previous_run_length = length
        self._previous_run_pairs = run_pairs

        self._current_run_kind = None
        self._current_run_length = 0
        self._current_run_pairs = []

    def process_line(self, line: str) -> bool:
        parsed = parse_hex_pair_line(line, self.cfg)
        if parsed is None:
            return False

        left, right, kind = parsed
        phase, frame_number, phase_index = advance_contract_tracker(self._tracker, kind)
        self._print_pair(left, right, kind, phase, frame_number, phase_index)

        self.kind_counts[kind] += 1
        if kind == "tag_mismatch":
            self.mismatch_count += 1
        if kind == "packet_index_mismatch":
            self.packet_index_mismatch_count += 1
        if kind == "unknown_tag":
            self.unknown_count += 1
        self.reserved_nonzero_words += int(left.reserved_nonzero) + int(right.reserved_nonzero)

        if self._current_run_kind is None:
            self._current_run_kind = kind
        elif kind != self._current_run_kind:
            self.transition_counts[f"{self._current_run_kind}->{kind}"] += 1
            self._finish_current_run()
            self._current_run_kind = kind

        self._current_run_length += 1
        self._current_run_pairs.append((left.payload, right.payload))
        self.pair_count += 1
        return True

    def finish(self) -> TransportSummary:
        self._finish_current_run()
        run_hist = {
            kind: dict(counter.most_common())
            for kind, counter in self.run_hist_by_kind.items()
            if counter
        }
        return TransportSummary(
            pair_count=self.pair_count,
            kind_counts=dict(self.kind_counts.most_common()),
            transition_counts=dict(self.transition_counts.most_common()),
            run_hist_by_kind=run_hist,
            mismatch_count=self.mismatch_count,
            packet_index_mismatch_count=self.packet_index_mismatch_count,
            unknown_count=self.unknown_count,
            reserved_nonzero_words=self.reserved_nonzero_words,
            bfpexp_to_fft_examples=tuple(self.bfpexp_to_fft_examples[:32]),
            frame_reports=tuple(self.frame_reports),
        )


def analyze_tagged_hex_stream(
    lines: Sequence[str] | TextIO,
    cfg: TaggedTransportConfig,
    *,
    print_limit: int = 1024,
    printer: Optional[Callable[[str], None]] = None,
) -> TransportSummary:
    monitor = TransportMonitor(cfg, print_limit=print_limit, printer=printer)
    for line in lines:
        monitor.process_line(line)
    return monitor.finish()


def _stderr_pump(stderr: TextIO, printer: Callable[[str], None]) -> None:
    for line in stderr:
        stripped = line.rstrip()
        if stripped:
            printer(f"[alsa_log] {stripped}")


def _legacy_capture_cmd(binary_path: str, device: str, rate: int, read_frames: int) -> list[str]:
    return [binary_path, device, str(rate), str(read_frames)]


def _run_live_capture(
    *,
    binary_path: str,
    device: str,
    rate: int,
    read_frames: int,
    capture_seconds: float,
    max_pairs: int,
    monitor: TransportMonitor,
    printer: Callable[[str], None],
) -> TransportSummary:
    cmd = _legacy_capture_cmd(binary_path, device, rate, read_frames)
    printer(f"Starting capture: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stderr_thread = None
    if proc.stderr is not None:
        stderr_thread = threading.Thread(target=_stderr_pump, args=(proc.stderr, printer), daemon=True)
        stderr_thread.start()

    deadline = None if capture_seconds <= 0 else (time.monotonic() + capture_seconds)
    try:
        assert proc.stdout is not None
        while True:
            if max_pairs > 0 and monitor.pair_count >= max_pairs:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break

            ready, _, _ = select.select([proc.stdout], [], [], 0.2)
            if not ready:
                if proc.poll() is not None:
                    break
                continue

            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            monitor.process_line(line)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        if stderr_thread is not None:
            stderr_thread.join(timeout=1.0)

    return monitor.finish()


def _print_summary(summary: TransportSummary, printer: Callable[[str], None]) -> None:
    printer("")
    printer("Summary")
    printer(
        f"pairs={summary.pair_count} "
        f"tag_mismatches={summary.mismatch_count} "
        f"packet_index_mismatches={summary.packet_index_mismatch_count} "
        f"unknown={summary.unknown_count} "
        f"reserved_nonzero_words={summary.reserved_nonzero_words}"
    )
    printer(f"kind_counts={summary.kind_counts}")
    printer(f"transition_counts={summary.transition_counts}")
    printer(f"run_hist_by_kind={summary.run_hist_by_kind}")
    printer(f"bfpexp_to_fft_examples={list(summary.bfpexp_to_fft_examples)}")
    if not summary.frame_reports:
        printer("full_fft_frames=0")
        return
    printer(f"full_fft_frames={len(summary.frame_reports)}")
    for report in summary.frame_reports:
        printer(
            "frame="
            f"{report.frame_index:03d} bfpexp={report.bfpexp_value} "
            f"bfpexp_run={report.bfpexp_run_length} fft_run={report.fft_run_length} "
            f"peak_bin={report.peak_bin} peak_mag={report.peak_magnitude:.3f} "
            f"shift_from_prev={report.shift_from_prev}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the raw tagged I2S transport and print live real/imag hex pairs from ~/Desktop/alsa_log."
    )
    parser.add_argument("--capture-binary", default=DEFAULT_CAPTURE_BINARY, help="Legacy hex capture helper (default: ~/Desktop/alsa_log)")
    parser.add_argument("-D", "--device", default=DEFAULT_CAPTURE_DEVICE, help="ALSA capture device")
    parser.add_argument("-r", "--rate", type=int, default=DEFAULT_CAPTURE_RATE_HZ, help="Capture rate in Hz")
    parser.add_argument("--read-frames", type=int, default=DEFAULT_CAPTURE_READ_FRAMES, help="Frames per ALSA read in the legacy helper")
    parser.add_argument("--seconds", type=float, default=4.0, help="Live capture duration in seconds (0 means until Ctrl+C)")
    parser.add_argument("--max-pairs", type=int, default=0, help="Stop after N decoded stereo pairs (0 means no pair limit)")
    parser.add_argument("--print-limit", type=int, default=1024, help="Print at most N raw hex pairs to stdout (0 prints all)")
    parser.add_argument("--input-file", help="Analyze an existing hex dump instead of launching ~/Desktop/alsa_log")
    parser.add_argument("--frame-bins", type=int, default=512, help="Expected FFT-tagged pairs per valid burst")
    parser.add_argument("--useful-bins", type=int, default=256, help="Useful FFT bins used in peak/shift analysis")
    parser.add_argument("--packet-index-shift", type=int, default=DEFAULT_PACKET_INDEX_SHIFT, help="Bit shift of the packet-index field")
    parser.add_argument("--packet-index-bits", type=int, default=DEFAULT_PACKET_INDEX_BITS, help="Packet-index width in bits")
    parser.add_argument("--fft-packet-index-base", type=int, default=DEFAULT_FFT_PACKET_INDEX_BASE, help="First packet index used by FFT payload words")
    parser.add_argument("--tag-shift", type=int, default=DEFAULT_TAG_SHIFT, help="Bit shift of the in-band tag")
    parser.add_argument("--tag-mask", type=lambda value: int(value, 0), default=DEFAULT_TAG_MASK, help="Bitmask of the in-band tag")
    parser.add_argument("--payload-bits", type=int, default=DEFAULT_PAYLOAD_BITS, help="Signed payload width inside each 32-bit word")
    parser.add_argument("--tag-idle", type=int, default=DEFAULT_TAG_IDLE, help="Tag value used for idle words")
    parser.add_argument("--tag-bfpexp", type=int, default=DEFAULT_TAG_BFPEXP, help="Tag value used for BFPEXP words")
    parser.add_argument("--tag-fft", type=int, default=DEFAULT_TAG_FFT, help="Tag value used for FFT words")
    parser.add_argument("--bfpexp-hold-pairs", type=int, default=128, help="Expected BFPEXP preamble length before a valid FFT burst")
    parser.add_argument("--allow-fft-without-bfpexp", action="store_true", help="Let the contract tracker accept FFT startup without a BFPEXP preamble")
    parser.add_argument("--loss-tolerance-pairs", type=int, default=0, help="Gap tolerance used only by the on-screen contract tracker")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.frame_bins <= 0:
        parser.error("--frame-bins must be positive")
    if args.frame_bins > args.fft_packet_index_base:
        parser.error("--frame-bins must fit inside the FFT packet-index range")
    if args.packet_index_bits <= 0:
        parser.error("--packet-index-bits must be positive")

    cfg = TaggedTransportConfig(
        frame_bins=args.frame_bins,
        useful_bins=args.useful_bins,
        packet_index_shift=args.packet_index_shift,
        packet_index_bits=args.packet_index_bits,
        fft_packet_index_base=args.fft_packet_index_base,
        tag_shift=args.tag_shift,
        tag_mask=args.tag_mask,
        payload_bits=args.payload_bits,
        tag_idle=args.tag_idle,
        tag_bfpexp=args.tag_bfpexp,
        tag_fft=args.tag_fft,
        bfpexp_pairs_required=args.bfpexp_hold_pairs,
        allow_fft_without_bfpexp=bool(args.allow_fft_without_bfpexp),
        loss_tolerance_pairs=args.loss_tolerance_pairs,
    )

    printer = print
    monitor = TransportMonitor(cfg, print_limit=args.print_limit, printer=printer)

    if args.input_file:
        input_path = Path(args.input_file).expanduser()
        with input_path.open("r", encoding="ascii", errors="replace") as handle:
            summary = analyze_tagged_hex_stream(
                handle,
                cfg,
                print_limit=args.print_limit,
                printer=printer,
            )
    else:
        binary_path = str(Path(args.capture_binary).expanduser())
        if not os.path.isfile(binary_path):
            parser.error(f"capture helper not found: {binary_path}")
        summary = _run_live_capture(
            binary_path=binary_path,
            device=args.device,
            rate=args.rate,
            read_frames=args.read_frames,
            capture_seconds=args.seconds,
            max_pairs=args.max_pairs,
            monitor=monitor,
            printer=printer,
        )

    _print_summary(summary, printer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
