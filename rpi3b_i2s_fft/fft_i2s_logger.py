import argparse
import collections
import csv
import os
import signal
import sys
import time
from typing import Optional

import numpy as np

try:
    from .fpga_fft_adapter import DEFAULT_BFPEXP_HOLD_PAIRS
    from .i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_RATE_HZ,
        TaggedI2SRealigner,
        build_arecord_cmd,
        read_exactly,
        resolve_audio_device,
        start_arecord_process,
        stop_process,
        trim_incomplete_frames,
    )
except ImportError:
    from fpga_fft_adapter import DEFAULT_BFPEXP_HOLD_PAIRS
    from i2s_stream import (
        AUTO_AUDIO_DEVICE,
        DEFAULT_CAPTURE_RATE_HZ,
        TaggedI2SRealigner,
        build_arecord_cmd,
        read_exactly,
        resolve_audio_device,
        start_arecord_process,
        stop_process,
        trim_incomplete_frames,
    )


DEFAULT_AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE") or AUTO_AUDIO_DEVICE
DEFAULT_LOGGER_CHUNK_FRAMES = 4096
DEFAULT_CSV_FLUSH_EVERY_CHUNKS = 8


def format_i32_hex(value: int) -> str:
    return f"0x{int(value) & 0xFFFFFFFF:08X}"


def decode_tagged_word(
    value: int,
    *,
    tag_shift: int,
    tag_mask: int,
    payload_bits: int,
) -> dict[str, object]:
    uword = int(value) & 0xFFFFFFFF
    tag = (uword >> tag_shift) & tag_mask

    payload_mask = (1 << payload_bits) - 1
    payload = uword & payload_mask
    sign_bit = 1 << (payload_bits - 1)
    if payload & sign_bit:
        payload -= 1 << payload_bits

    reserved_width = max(0, tag_shift - payload_bits)
    reserved = 0
    if reserved_width > 0:
        reserved = (uword >> payload_bits) & ((1 << reserved_width) - 1)

    return {
        "tag": int(tag),
        "payload": int(payload),
        "reserved": int(reserved),
        "reserved_nonzero": bool(reserved),
    }


def classify_tagged_pair(
    left_tag: int,
    right_tag: int,
    *,
    tag_idle: int,
    tag_bfpexp: int,
    tag_fft: int,
) -> str:
    if left_tag != right_tag:
        return "tag_mismatch"
    if left_tag == tag_idle:
        return "idle"
    if left_tag == tag_bfpexp:
        return "bfpexp"
    if left_tag == tag_fft:
        return "fft"
    return "unknown_tag"


def create_contract_tracker(
    *,
    frame_bins: int,
    bfpexp_hold_pairs: int,
    allow_fft_without_bfpexp: bool,
) -> dict[str, object]:
    return {
        "frame_bins": int(frame_bins),
        "bfpexp_hold_pairs": int(bfpexp_hold_pairs),
        "allow_fft_without_bfpexp": bool(allow_fft_without_bfpexp),
        "frame_number": 0,
        "bfpexp_run": 0,
        "fft_index": 0,
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
                tracker["bootstrapped"] = False
                tracker["frame_number"] = frame_number + 1
            return phase, frame_number, fft_index

        tracker["inside_fft"] = False
        tracker["fft_index"] = 0
        tracker["bootstrapped"] = False
        if kind == "bfpexp":
            tracker["bfpexp_run"] = 1
            return "protocol_reset_bfpexp", frame_number, 0
        tracker["bfpexp_run"] = 0
        return f"protocol_reset_{kind}", frame_number, -1

    if kind == "idle":
        if int(tracker["bfpexp_run"]) != 0:
            tracker["bfpexp_run"] = 0
        return "search_idle", frame_number, -1

    if kind == "bfpexp":
        bfpexp_index = int(tracker["bfpexp_run"])
        tracker["bfpexp_run"] = bfpexp_index + 1
        return "bfpexp_preamble", frame_number, bfpexp_index

    if kind == "fft":
        bfpexp_run = int(tracker["bfpexp_run"])
        bootstrap = bfpexp_run == 0 and bool(tracker["allow_fft_without_bfpexp"])
        if (not bootstrap) and (bfpexp_run < int(tracker["bfpexp_hold_pairs"])):
            tracker["bfpexp_run"] = 0
            return "protocol_wait_bfpexp", frame_number, -1

        tracker["inside_fft"] = True
        tracker["bootstrapped"] = bootstrap
        tracker["fft_index"] = 1
        phase = "fft_frame_bootstrap" if bootstrap else "fft_frame"
        if int(tracker["frame_bins"]) == 1:
            tracker["inside_fft"] = False
            tracker["fft_index"] = 0
            tracker["bfpexp_run"] = 0
            tracker["bootstrapped"] = False
            tracker["frame_number"] = frame_number + 1
        return phase, frame_number, 0

    tracker["bfpexp_run"] = 0
    return f"protocol_reset_{kind}", frame_number, -1


class MirroredPairNormalizer:
    def __init__(self):
        self._preferred_by_signature: dict[tuple[int, int], tuple[int, int]] = {}

    def normalize(self, stereo: np.ndarray) -> np.ndarray:
        stereo_i32 = np.asarray(stereo, dtype=np.int32)
        if stereo_i32.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        if stereo_i32.ndim != 2 or stereo_i32.shape[1] != 2:
            stereo_i32 = stereo_i32.reshape(-1, 2)

        stereo_u32 = stereo_i32.astype(np.uint32, copy=False)
        counts = collections.Counter((int(left), int(right)) for left, right in stereo_u32)
        chunk_preference: dict[tuple[int, int], tuple[int, int]] = {}

        for (left, right), count in counts.items():
            if left == right:
                continue

            reversed_pair = (right, left)
            key = (left, right) if left < right else (right, left)
            preferred = self._preferred_by_signature.get(key)
            if preferred is None:
                reversed_count = counts.get(reversed_pair, 0)
                if reversed_count > count:
                    preferred = reversed_pair
                else:
                    preferred = (left, right)
                self._preferred_by_signature[key] = preferred
            chunk_preference[key] = preferred

        if not chunk_preference:
            return stereo_i32.copy()

        normalized = stereo_u32.copy()
        for row in normalized:
            left = int(row[0])
            right = int(row[1])
            if left == right:
                continue
            key = (left, right) if left < right else (right, left)
            preferred = chunk_preference.get(key)
            if preferred is None:
                continue
            if (left, right) != preferred:
                row[0], row[1] = row[1], row[0]

        return normalized.view(np.int32)


def decode_stereo_frames(raw: bytes) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.int32)
    if data.size < 2 or (data.size % 2) != 0:
        return np.empty((0, 2), dtype=np.int32)
    return data.reshape(-1, 2)


def write_csv_rows(
    writer: csv.writer,
    stereo: np.ndarray,
    seq_start: int,
    *,
    tag_shift: int,
    tag_mask: int,
    payload_bits: int,
    tag_idle: int,
    tag_bfpexp: int,
    tag_fft: int,
    contract_tracker: Optional[dict[str, object]] = None,
    timestamp_ns_fn=time.time_ns,
) -> int:
    seq = seq_start
    rows = []
    for row in stereo:
        left = decode_tagged_word(
            int(row[0]),
            tag_shift=tag_shift,
            tag_mask=tag_mask,
            payload_bits=payload_bits,
        )
        right = decode_tagged_word(
            int(row[1]),
            tag_shift=tag_shift,
            tag_mask=tag_mask,
            payload_bits=payload_bits,
        )
        kind = classify_tagged_pair(
            int(left["tag"]),
            int(right["tag"]),
            tag_idle=tag_idle,
            tag_bfpexp=tag_bfpexp,
            tag_fft=tag_fft,
        )
        contract_phase = ""
        contract_frame = -1
        contract_index = -1
        if contract_tracker is not None:
            contract_phase, contract_frame, contract_index = advance_contract_tracker(contract_tracker, kind)
        rows.append(
            [
                timestamp_ns_fn(),
                seq,
                format_i32_hex(row[0]),
                format_i32_hex(row[1]),
                kind,
                contract_phase,
                contract_frame,
                contract_index,
                left["tag"],
                right["tag"],
                left["payload"],
                right["payload"],
                left["reserved"],
                right["reserved"],
                int(bool(left["reserved_nonzero"])),
                int(bool(right["reserved_nonzero"])),
            ]
        )
        seq = (seq + 1) & 0xFFFFFFFF
    writer.writerows(rows)
    return seq


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture I2S FFT stream and log raw real/imag pairs to CSV."
    )
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_AUDIO_DEVICE,
        help="ALSA capture device (default: $AUDIO_DEVICE if set, otherwise auto-detect)",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=DEFAULT_CAPTURE_RATE_HZ,
        help="Host-side ALSA sample rate in Hz (nominal wire rate is 48828.125 Hz)",
    )
    parser.add_argument("--frame-bins", type=int, default=512, help="Expected FFT-tagged pairs per burst")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_LOGGER_CHUNK_FRAMES,
        help="Frames read per chunk",
    )
    parser.add_argument("--csv", default="fft_capture.csv", help="Output CSV path")
    parser.add_argument(
        "--flush-every-chunks",
        type=int,
        default=DEFAULT_CSV_FLUSH_EVERY_CHUNKS,
        help="Flush CSV data every N chunks instead of every chunk",
    )
    parser.add_argument("--tag-shift", type=int, default=30, help="Bit shift of type tag in each 32-bit word")
    parser.add_argument("--tag-mask", type=lambda v: int(v, 0), default=0x3, help="Bitmask for type tag")
    parser.add_argument("--payload-bits", type=int, default=18, help="Signed payload width inside each word")
    parser.add_argument("--tag-idle", type=int, default=0, help="Tag value representing idle/no data")
    parser.add_argument("--tag-bfpexp", type=int, default=1, help="Tag value representing BFPEXP data")
    parser.add_argument("--tag-fft", type=int, default=2, help="Tag value representing FFT complex bins")
    parser.add_argument(
        "--bfpexp-hold-pairs",
        type=int,
        default=DEFAULT_BFPEXP_HOLD_PAIRS,
        help="Expected consecutive BFPEXP-tagged stereo pairs before each FFT burst",
    )
    parser.add_argument(
        "--allow-fft-without-bfpexp",
        action="store_true",
        help="Annotate FFT bursts as valid even if they start without a BFPEXP preamble",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.frame_bins <= 0:
        parser.error("--frame-bins must be positive")
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")
    if args.flush_every_chunks <= 0:
        parser.error("--flush-every-chunks must be positive")
    if args.payload_bits <= 0:
        parser.error("--payload-bits must be positive")
    if args.bfpexp_hold_pairs <= 0:
        parser.error("--bfpexp-hold-pairs must be positive")

    try:
        device = resolve_audio_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    bytes_per_frame = 8  # 2 channels x int32
    chunk_bytes = args.chunk_frames * bytes_per_frame

    seq = 0
    chunk_index = 0
    stop = False
    realigner = TaggedI2SRealigner(
        tag_shift=args.tag_shift,
        tag_mask=args.tag_mask,
        payload_bits=args.payload_bits,
        tag_idle=args.tag_idle,
        tag_bfpexp=args.tag_bfpexp,
        tag_fft=args.tag_fft,
        preferred_swap_channels=True,
    )
    normalizer = MirroredPairNormalizer()
    contract_tracker = create_contract_tracker(
        frame_bins=args.frame_bins,
        bfpexp_hold_pairs=args.bfpexp_hold_pairs,
        allow_fft_without_bfpexp=args.allow_fft_without_bfpexp,
    )

    def handle_stop(_sig: int, _frame: Optional[object]) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    cmd = build_arecord_cmd(device, args.rate)
    print("Using ALSA capture device:", device, flush=True)
    print("Starting:", " ".join(cmd), flush=True)
    print("Logging CSV to:", args.csv, flush=True)

    try:
        proc = start_arecord_process(device, args.rate)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        with open(args.csv, "w", newline="", encoding="ascii", buffering=1024 * 1024) as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(
                [
                    "timestamp_ns",
                    "seq",
                    "left_hex",
                    "right_hex",
                    "kind",
                    "contract_phase",
                    "contract_frame",
                    "contract_index",
                    "left_tag",
                    "right_tag",
                    "left_payload",
                    "right_payload",
                    "left_reserved",
                    "right_reserved",
                    "left_reserved_nonzero",
                    "right_reserved_nonzero",
                ]
            )

            while not stop:
                if proc.stdout is None:
                    raise RuntimeError("arecord stdout pipe is unavailable")

                raw = trim_incomplete_frames(read_exactly(proc.stdout, chunk_bytes), bytes_per_frame)
                if not raw:
                    if proc.poll() is not None:
                        print(f"arecord exited with code {proc.returncode}", file=sys.stderr)
                        return 1

                    continue

                stereo = decode_stereo_frames(raw)
                if stereo.size == 0:
                    continue

                stereo = realigner.push_pairs(stereo)
                if stereo.size == 0:
                    continue
                stereo = normalizer.normalize(stereo)

                seq = write_csv_rows(
                    writer,
                    stereo,
                    seq,
                    tag_shift=args.tag_shift,
                    tag_mask=args.tag_mask,
                    payload_bits=args.payload_bits,
                    tag_idle=args.tag_idle,
                    tag_bfpexp=args.tag_bfpexp,
                    tag_fft=args.tag_fft,
                    contract_tracker=contract_tracker,
                )
                chunk_index += 1
                if (chunk_index % args.flush_every_chunks) == 0:
                    f_csv.flush()

            f_csv.flush()

    finally:
        stop_process(proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
