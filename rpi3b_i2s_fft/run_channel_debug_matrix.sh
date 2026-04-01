#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a small matrix of passive channel-debug replays for the FPGA I2S FFT stream.

The script stores:
- one shared raw capture of the live I2S stream,
- one chunk-index JSONL aligned with that raw capture,
- one JSONL log per scenario,
- the exact command used for each scenario,
- a machine-readable TSV summary,
- a session info file you can commit alongside the logs.

Default scenarios:
1. strict_tags_shift30
2. relaxed_tags_shift30
3. relaxed_tags_shift29

Usage:
  ./run_channel_debug_matrix.sh [options]

Options:
  -D, --device DEV             ALSA device passed to analyzer (default: auto)
  -r, --rate HZ                Sample rate (default: 48828)
      --frame-bins N           FFT bins per frame (default: 512)
      --useful-bins N          Useful bins kept by analyzer config (default: 256)
      --seconds S              Shared raw capture time (default: 12)
      --output-dir DIR         Output directory (default: ./debug_runs/<timestamp>)
      --gpio-chip PATH         GPIO chip path (default: /dev/gpiochip0)
      --bfpexp-flag-line N     Optional GPIO input line captured in debug logs
      --flag-active-low        Mark BFPEXP flag as active-low
      --wait-low-level         Use low-level wait semantics in analyzer config
      --tag-shift N            Nominal tag shift (default: 30)
      --tag-mask N             Tag mask, accepts 0x... (default: 0x3)
      --payload-bits N         Signed payload width (default: 18)
      --tag-idle N             Idle tag value (default: 0)
      --tag-bfpexp N           BFPEXP tag value (default: 1)
      --tag-fft N              FFT tag value (default: 2)
      --extra-tag-shifts LIST  Extra relaxed probe shifts, space or comma separated
                               (default: 29)
      --chunk-pairs N          Raw stereo pairs captured/replayed per chunk (default: 1024)
      --preview-pairs N        Preview pairs stored per chunk (default: 12)
      --python BIN             Python interpreter to use
      --dry-run                Print commands and generate manifest without running captures
  -h, --help                   Show this help

Examples:
  ./run_channel_debug_matrix.sh --seconds 8
  ./run_channel_debug_matrix.sh --device hw:2,0 --bfpexp-flag-line 23
  ./run_channel_debug_matrix.sh --output-dir debug_runs/pi_after_i2s_fix
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

DEVICE="${AUDIO_DEVICE:-auto}"
RATE=48828
FRAME_BINS=512
USEFUL_BINS=256
CAPTURE_SECONDS=12
OUTPUT_DIR=""
GPIO_CHIP="/dev/gpiochip0"
BFPEXP_FLAG_LINE=""
FLAG_ACTIVE_LOW=0
WAIT_LOW_LEVEL=0
TAG_SHIFT=30
TAG_MASK="0x3"
PAYLOAD_BITS=18
TAG_IDLE=0
TAG_BFPEXP=1
TAG_FFT=2
EXTRA_TAG_SHIFTS_RAW="29"
CHUNK_PAIRS=1024
PREVIEW_PAIRS=12
PYTHON_BIN="${DEFAULT_PYTHON}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -D|--device)
      DEVICE="$2"
      shift 2
      ;;
    -r|--rate)
      RATE="$2"
      shift 2
      ;;
    --frame-bins)
      FRAME_BINS="$2"
      shift 2
      ;;
    --useful-bins)
      USEFUL_BINS="$2"
      shift 2
      ;;
    --seconds)
      CAPTURE_SECONDS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpio-chip)
      GPIO_CHIP="$2"
      shift 2
      ;;
    --bfpexp-flag-line)
      BFPEXP_FLAG_LINE="$2"
      shift 2
      ;;
    --flag-active-low)
      FLAG_ACTIVE_LOW=1
      shift
      ;;
    --wait-low-level)
      WAIT_LOW_LEVEL=1
      shift
      ;;
    --tag-shift)
      TAG_SHIFT="$2"
      shift 2
      ;;
    --tag-mask)
      TAG_MASK="$2"
      shift 2
      ;;
    --payload-bits)
      PAYLOAD_BITS="$2"
      shift 2
      ;;
    --tag-idle)
      TAG_IDLE="$2"
      shift 2
      ;;
    --tag-bfpexp)
      TAG_BFPEXP="$2"
      shift 2
      ;;
    --tag-fft)
      TAG_FFT="$2"
      shift 2
      ;;
    --extra-tag-shifts)
      EXTRA_TAG_SHIFTS_RAW="$2"
      shift 2
      ;;
    --chunk-pairs)
      CHUNK_PAIRS="$2"
      shift 2
      ;;
    --preview-pairs)
      PREVIEW_PAIRS="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
fi

ANALYZER_PATH="${SCRIPT_DIR}/analyzer_from_fpga_fft.py"
if [[ ! -f "${ANALYZER_PATH}" ]]; then
  echo "Analyzer not found at ${ANALYZER_PATH}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${SCRIPT_DIR}/debug_runs/${timestamp}"
fi
mkdir -p "${OUTPUT_DIR}"

MANIFEST_PATH="${OUTPUT_DIR}/session_info.txt"
SUMMARY_PATH="${OUTPUT_DIR}/scenario_summary.tsv"
README_PATH="${OUTPUT_DIR}/README.txt"
COMMANDS_PATH="${OUTPUT_DIR}/replay_commands.sh"
RAW_CAPTURE_PATH="${OUTPUT_DIR}/channel_capture.raw"
RAW_INDEX_PATH="${OUTPUT_DIR}/channel_capture.index.jsonl"

normalize_shift_list() {
  local raw="$1"
  raw="${raw//,/ }"
  raw="$(echo "${raw}" | xargs || true)"
  printf '%s\n' "${raw}"
}

EXTRA_TAG_SHIFTS_NORM="$(normalize_shift_list "${EXTRA_TAG_SHIFTS_RAW}")"
read -r -a EXTRA_TAG_SHIFTS <<<"${EXTRA_TAG_SHIFTS_NORM}"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo
  echo "# Commands generated by run_channel_debug_matrix.sh"
} > "${COMMANDS_PATH}"
chmod +x "${COMMANDS_PATH}"

{
  echo "Channel debug capture set"
  echo "generated_at=$(date -Is)"
  echo "script=${SCRIPT_DIR}/run_channel_debug_matrix.sh"
  echo "analyzer=${ANALYZER_PATH}"
  echo "python=${PYTHON_BIN}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "device=${DEVICE}"
  echo "rate=${RATE}"
  echo "frame_bins=${FRAME_BINS}"
  echo "useful_bins=${USEFUL_BINS}"
  echo "capture_seconds=${CAPTURE_SECONDS}"
  echo "gpio_chip=${GPIO_CHIP}"
  echo "bfpexp_flag_line=${BFPEXP_FLAG_LINE:-none}"
  echo "flag_active_low=${FLAG_ACTIVE_LOW}"
  echo "wait_low_level=${WAIT_LOW_LEVEL}"
  echo "tag_shift=${TAG_SHIFT}"
  echo "tag_mask=${TAG_MASK}"
  echo "payload_bits=${PAYLOAD_BITS}"
  echo "tag_idle=${TAG_IDLE}"
  echo "tag_bfpexp=${TAG_BFPEXP}"
  echo "tag_fft=${TAG_FFT}"
  echo "extra_tag_shifts=${EXTRA_TAG_SHIFTS_NORM:-none}"
  echo "chunk_pairs=${CHUNK_PAIRS}"
  echo "preview_pairs=${PREVIEW_PAIRS}"
  echo "raw_capture=${RAW_CAPTURE_PATH}"
  echo "raw_index=${RAW_INDEX_PATH}"
  echo "dry_run=${DRY_RUN}"
  echo
  echo "[git]"
  (cd "${SCRIPT_DIR}" && git rev-parse HEAD 2>/dev/null | sed 's/^/submodule_head=/' ) || true
  (cd "${SCRIPT_DIR}" && git status --short 2>/dev/null) || true
  echo
  echo "[system]"
  uname -a || true
  echo
  echo "[arecord -l]"
  arecord -l 2>&1 || true
} > "${MANIFEST_PATH}"

cat > "${README_PATH}" <<EOF
This directory was generated by run_channel_debug_matrix.sh.

Files:
- channel_capture.raw: one shared raw S32_LE stereo capture from the FPGA I2S stream
- channel_capture.index.jsonl: chunk timestamps, pair counts, and sampled BFPEXP flag state
- all scenario JSONL logs replay the same raw capture for apples-to-apples comparison
- session_info.txt: environment, git state, ALSA listing, and matrix parameters
- replay_commands.sh: exact commands used for each scenario
- scenario_summary.tsv: compact summary extracted from the JSONL logs
- *.jsonl: one passive channel-debug analysis per scenario

Recommended commit:
1. Commit this whole directory.
2. Mention which FPGA bitstream and physical wiring were used.
3. If there is an external BFPEXP GPIO wire, include the line number used here.
4. Keep the raw capture and index together so we can replay new hypotheses at home.
EOF

printf '%s\n' \
  "scenario	status	log_file	duration_seconds	total_pairs	idle	bfpexp	fft	tag_mismatch	other	max_fft_run	reserved_nonzero_words	flag_high_chunks	flag_low_chunks	flag_unknown_chunks	interrupted" \
  > "${SUMMARY_PATH}"

COMMON_ARGS=(
  "-D" "${DEVICE}"
  "-r" "${RATE}"
  "--frame-bins" "${FRAME_BINS}"
  "--useful-bins" "${USEFUL_BINS}"
  "--gpio-chip" "${GPIO_CHIP}"
  "--debug-capture-seconds" "${CAPTURE_SECONDS}"
  "--debug-chunk-pairs" "${CHUNK_PAIRS}"
  "--debug-preview-pairs" "${PREVIEW_PAIRS}"
  "--tag-mask" "${TAG_MASK}"
  "--payload-bits" "${PAYLOAD_BITS}"
  "--tag-idle" "${TAG_IDLE}"
  "--tag-bfpexp" "${TAG_BFPEXP}"
  "--tag-fft" "${TAG_FFT}"
)

if [[ -n "${BFPEXP_FLAG_LINE}" ]]; then
  COMMON_ARGS+=("--bfpexp-flag-line" "${BFPEXP_FLAG_LINE}")
fi
if [[ "${FLAG_ACTIVE_LOW}" -eq 1 ]]; then
  COMMON_ARGS+=("--flag-active-low")
fi
if [[ "${WAIT_LOW_LEVEL}" -eq 1 ]]; then
  COMMON_ARGS+=("--wait-low-level")
fi

extract_summary() {
  local scenario_name="$1"
  local status="$2"
  local log_path="$3"

  "${PYTHON_BIN}" - "${scenario_name}" "${status}" "${log_path}" >> "${SUMMARY_PATH}" <<'PY'
import json
import pathlib
import sys

scenario = sys.argv[1]
status = sys.argv[2]
log_path = pathlib.Path(sys.argv[3])

summary = {}
if log_path.exists():
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "summary":
                summary = payload

kind_counts = summary.get("kind_counts", {})
max_run = summary.get("max_run_by_kind", {})

row = [
    scenario,
    status,
    log_path.name,
    str(summary.get("duration_seconds", "")),
    str(summary.get("total_pairs", "")),
    str(kind_counts.get("idle", "")),
    str(kind_counts.get("bfpexp", "")),
    str(kind_counts.get("fft", "")),
    str(kind_counts.get("tag_mismatch", "")),
    str(kind_counts.get("other", "")),
    str(max_run.get("fft", "")),
    str(summary.get("reserved_nonzero_words", "")),
    str(summary.get("flag_high_chunks", "")),
    str(summary.get("flag_low_chunks", "")),
    str(summary.get("flag_unknown_chunks", "")),
    str(summary.get("interrupted", "")),
]
print("\t".join(row))
PY
}

run_case() {
  local scenario_name="$1"
  shift

  local log_path="${OUTPUT_DIR}/${scenario_name}.jsonl"
  local cmd=(
    "${PYTHON_BIN}"
    "${ANALYZER_PATH}"
    "${COMMON_ARGS[@]}"
    "--debug-channel-log" "${log_path}"
    "--debug-replay-raw" "${RAW_CAPTURE_PATH}"
    "--debug-raw-index" "${RAW_INDEX_PATH}"
    "$@"
  )

  {
    printf '# %s\n' "${scenario_name}"
    printf '%q ' "${cmd[@]}"
    printf '\n\n'
  } >> "${COMMANDS_PATH}"

  echo "[${scenario_name}] log => ${log_path}"
  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%s\n' "${scenario_name}	skipped	$(basename "${log_path}")								" >> "${SUMMARY_PATH}"
    return 0
  fi

  if "${cmd[@]}"; then
    extract_summary "${scenario_name}" "ok" "${log_path}"
  else
    extract_summary "${scenario_name}" "failed" "${log_path}"
    return 1
  fi
}

run_shared_capture() {
  local cmd=(
    "${PYTHON_BIN}"
    "${ANALYZER_PATH}"
    "${COMMON_ARGS[@]}"
    "--debug-raw-capture" "${RAW_CAPTURE_PATH}"
    "--debug-raw-index" "${RAW_INDEX_PATH}"
  )

  {
    printf '# %s\n' "shared_raw_capture"
    printf '%q ' "${cmd[@]}"
    printf '\n\n'
  } >> "${COMMANDS_PATH}"

  echo "[shared_raw_capture] raw => ${RAW_CAPTURE_PATH}"
  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  "${cmd[@]}"
}

scenario_failures=0

if ! run_shared_capture; then
  echo "Shared raw capture failed." >&2
  exit 1
fi

if ! run_case \
  "strict_tags_shift${TAG_SHIFT}" \
  "--use-i2s-tags" \
  "--tag-shift" "${TAG_SHIFT}"; then
  scenario_failures=$((scenario_failures + 1))
fi

if ! run_case \
  "relaxed_tags_shift${TAG_SHIFT}" \
  "--use-i2s-tags" \
  "--tag-shift" "${TAG_SHIFT}" \
  "--allow-fft-without-bfpexp"; then
  scenario_failures=$((scenario_failures + 1))
fi

for extra_shift in "${EXTRA_TAG_SHIFTS[@]}"; do
  if [[ -z "${extra_shift}" || "${extra_shift}" == "${TAG_SHIFT}" ]]; then
    continue
  fi

  if ! run_case \
    "relaxed_tags_shift${extra_shift}" \
    "--use-i2s-tags" \
    "--tag-shift" "${extra_shift}" \
    "--allow-fft-without-bfpexp"; then
    scenario_failures=$((scenario_failures + 1))
  fi
done

echo
echo "Artifacts written to ${OUTPUT_DIR}"
echo "Summary: ${SUMMARY_PATH}"
echo "Replay commands: ${COMMANDS_PATH}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run complete."
  exit 0
fi

if [[ "${scenario_failures}" -ne 0 ]]; then
  echo "One or more debug scenarios failed." >&2
  exit 1
fi

echo "All debug scenarios completed."
