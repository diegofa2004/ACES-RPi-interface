#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly OVERLAY_NAME="fpga-i2s-rx-32x2-slave"
readonly OVERLAY_DTS="${SCRIPT_DIR}/${OVERLAY_NAME}-overlay.dts"
readonly OVERLAY_DTBO="${SCRIPT_DIR}/${OVERLAY_NAME}.dtbo"
readonly MODULE_NAME="snd-soc-fpgafft-codec"
readonly MODULE_KO="${SCRIPT_DIR}/${MODULE_NAME}.ko"
readonly HOST_RATE_HZ="48828"
readonly WIRE_RATE_HZ="48828.125"

SKIP_BUILD=0
ENSURE_DTPARAM=1

log() {
  printf '[fpgafft-install] %s\n' "$*"
}

die() {
  printf '[fpgafft-install] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  sudo ./install_fpgafft_overlay.sh [--skip-build] [--skip-dtparam]

Options:
  --skip-build    Reuse existing .ko/.dtbo artifacts instead of rebuilding them.
  --skip-dtparam  Do not enforce dtparam=i2s=on in config.txt.
  -h, --help      Show this help message.
EOF
}

find_config_file() {
  if [[ -f /boot/firmware/config.txt ]]; then
    printf '/boot/firmware/config.txt\n'
    return 0
  fi

  if [[ -f /boot/config.txt ]]; then
    printf '/boot/config.txt\n'
    return 0
  fi

  return 1
}

find_overlay_dir() {
  local config_file="$1"

  case "${config_file}" in
    /boot/firmware/config.txt)
      printf '/boot/firmware/overlays\n'
      ;;
    /boot/config.txt)
      printf '/boot/overlays\n'
      ;;
    *)
      return 1
      ;;
  esac
}

find_kernel_build_dir() {
  local kernel_release
  kernel_release="$(uname -r)"
  printf '/lib/modules/%s/build\n' "${kernel_release}"
}

backup_file() {
  local source_file="$1"
  local timestamp="$2"
  local backup_path="${source_file}.bak.${timestamp}"

  cp "${source_file}" "${backup_path}"
  log "Backup created: ${backup_path}"
}

ensure_line() {
  local config_file="$1"
  local line="$2"

  if grep -Fqx "${line}" "${config_file}"; then
    log "Already present in config.txt: ${line}"
    return 0
  fi

  printf '\n%s\n' "${line}" >> "${config_file}"
  log "Added to config.txt: ${line}"
}

build_artifacts() {
  local kernel_build_dir="$1"

  command -v make >/dev/null 2>&1 || die "'make' was not found. Install build-essential."
  command -v dtc >/dev/null 2>&1 || die "'dtc' was not found. Install device-tree-compiler."
  [[ -d "${kernel_build_dir}" ]] || die "Kernel build directory not found: ${kernel_build_dir}"

  log "Building module and overlay in ${SCRIPT_DIR}"
  make -C "${SCRIPT_DIR}" KERNEL_BUILD_DIR="${kernel_build_dir}" all
}

install_dtbo() {
  local overlay_dir="$1"
  local timestamp="$2"
  local destination="${overlay_dir}/${OVERLAY_NAME}.dtbo"

  [[ -f "${OVERLAY_DTBO}" ]] || die "Missing overlay artifact: ${OVERLAY_DTBO}"
  mkdir -p "${overlay_dir}"

  if [[ -f "${destination}" ]]; then
    backup_file "${destination}" "${timestamp}"
  fi

  install -m 0644 "${OVERLAY_DTBO}" "${destination}"
  log "Installed overlay: ${destination}"
}

install_module() {
  local timestamp="$1"
  local kernel_release
  local destination_dir
  local destination

  [[ -f "${MODULE_KO}" ]] || die "Missing module artifact: ${MODULE_KO}"

  kernel_release="$(uname -r)"
  destination_dir="/lib/modules/${kernel_release}/extra"
  destination="${destination_dir}/${MODULE_NAME}.ko"

  mkdir -p "${destination_dir}"

  if [[ -f "${destination}" ]]; then
    backup_file "${destination}" "${timestamp}"
  fi

  install -m 0644 "${MODULE_KO}" "${destination}"
  depmod -a "${kernel_release}"
  log "Installed module: ${destination}"
}

print_post_reboot_checklist() {
  cat <<EOF

Install complete.

Hardware contract:
- FPGA remains the physical I2S master.
- Pi remains the I2S slave/capture side.
- ALSA host open rate: ${HOST_RATE_HZ} Hz
- Physical wire rate: ${WIRE_RATE_HZ} Hz

Reboot the Raspberry Pi before validating:
  sudo reboot

Post-reboot checklist:
  arecord -l
  aplay -l
  pinctrl get 18
  pinctrl get 19
  pinctrl get 20
  pinctrl get 21
  dmesg -l err,warn
  arecord --dump-hw-params -D hw:X,Y

Raw word validation:
  cd /path/to/submodules/ACES-RPi-interface/rpi3b_i2s_fft
  gcc -O2 -Wall -Wextra -pthread -o alsa_logger alsa_logger.c -lasound
  ./alsa_logger --device hw:X,Y --rate ${HOST_RATE_HZ}

The next stage is to confirm stable hexadecimal 32-bit words before returning
to the Python parser and tagged-mode semantics.
EOF
}

main() {
  local config_file
  local overlay_dir
  local kernel_build_dir
  local timestamp

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-build)
        SKIP_BUILD=1
        ;;
      --skip-dtparam)
        ENSURE_DTPARAM=0
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
    shift
  done

  if [[ "${EUID}" -ne 0 ]]; then
    die "Run this script with sudo/root."
  fi

  [[ -f "${OVERLAY_DTS}" ]] || die "Overlay source not found: ${OVERLAY_DTS}"
  [[ -f "${SCRIPT_DIR}/${MODULE_NAME}.c" ]] || die "Codec source not found in ${SCRIPT_DIR}"

  config_file="$(find_config_file)" || die "Could not find /boot/firmware/config.txt or /boot/config.txt"
  overlay_dir="$(find_overlay_dir "${config_file}")" || die "Could not infer the overlays directory from ${config_file}"
  kernel_build_dir="$(find_kernel_build_dir)"
  timestamp="$(date +%Y%m%d_%H%M%S)"

  backup_file "${config_file}" "${timestamp}"

  if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    build_artifacts "${kernel_build_dir}"
  fi

  install_dtbo "${overlay_dir}" "${timestamp}"
  install_module "${timestamp}"

  if [[ "${ENSURE_DTPARAM}" -eq 1 ]]; then
    ensure_line "${config_file}" "dtparam=i2s=on"
  fi
  ensure_line "${config_file}" "dtoverlay=${OVERLAY_NAME}"

  log "Target config.txt: ${config_file}"
  log "Target overlays directory: ${overlay_dir}"
  print_post_reboot_checklist
}

main "$@"
