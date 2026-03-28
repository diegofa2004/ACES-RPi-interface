#!/usr/bin/env bash
set -euo pipefail

# Configure Raspberry Pi OS for I2S capture and install runtime dependencies.
# Usage:
#   sudo ./setup_rpi_i2s_fft.sh
# Optional environment vars:
#   For FPGA-master mode, choose an overlay compatible with external BCLK/LRCLK.
#   I2S_OVERLAY=googlevoicehat-soundcard
#   AUDIO_DEVICE=hw:2,0
#
# Default Raspberry Pi PCM/I2S GPIO mapping used by overlays:
#   GPIO18 -> PCM_CLK  (I2S BCLK)
#   GPIO19 -> PCM_FS   (I2S LRCLK/WS)
#   GPIO20 -> PCM_DIN  (I2S SD input)
#   GPIO21 -> PCM_DOUT (I2S SD output, optional for capture-only)
#
# These defaults come from the SoC PCM/I2S peripheral pinmux (ALT functions)
# selected by the device-tree overlay, not from explicit per-pin commands here.

I2S_OVERLAY="${I2S_OVERLAY:-googlevoicehat-soundcard}"
AUDIO_DEVICE="${AUDIO_DEVICE:-hw:2,0}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo."
  exit 1
fi

if [[ -f /boot/firmware/config.txt ]]; then
  CONFIG_FILE=/boot/firmware/config.txt
elif [[ -f /boot/config.txt ]]; then
  CONFIG_FILE=/boot/config.txt
else
  echo "Could not find boot config file."
  exit 1
fi

BACKUP_FILE="${CONFIG_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
cp "${CONFIG_FILE}" "${BACKUP_FILE}"
echo "Backup created at ${BACKUP_FILE}"

ensure_line() {
  local line="$1"
  if ! grep -Eq "^${line}$" "${CONFIG_FILE}"; then
    echo "${line}" >> "${CONFIG_FILE}"
    echo "Added: ${line}"
  else
    echo "Already present: ${line}"
  fi
}

ensure_line "dtparam=i2s=on"
ensure_line "dtoverlay=${I2S_OVERLAY}"
echo "Using I2S overlay: ${I2S_OVERLAY}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-pip python3-venv python3-gpiod alsa-utils

if [[ ! -d "${PROJECT_DIR}/.venv" ]]; then
  python3 -m venv "${PROJECT_DIR}/.venv"
fi

"${PROJECT_DIR}/.venv/bin/pip" install --upgrade pip
"${PROJECT_DIR}/.venv/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"

cat <<EOF

Setup completed.

Next steps:
1. Reboot the Raspberry Pi:
   sudo reboot
2. After reboot, list ALSA devices:
   arecord -l
3. Start the I2S FFT daemon (update -D device if needed):
  ${PROJECT_DIR}/.venv/bin/python ${PROJECT_DIR}/fft_i2s_daemon.py -D ${AUDIO_DEVICE} -r 48000
4. In another shell, read latest real/imag values:
   ${PROJECT_DIR}/.venv/bin/python ${PROJECT_DIR}/fft_i2s_client.py --watch

If your overlay creates a different card/device, set AUDIO_DEVICE accordingly.
If you are using FPGA as I2S master, ensure the selected overlay supports external BCLK/LRCLK input.
EOF
