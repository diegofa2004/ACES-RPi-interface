#!/usr/bin/env bash
set -euo pipefail

# Configure Raspberry Pi OS for SPI-based FFT capture and install runtime dependencies.
# Usage:
#   sudo ./setup_rpi_spi_fft.sh
#
# Optional environment vars:
#   SPI_DEVICE=/dev/spidev0.0
#   WINDOW_READY_LINE=23
#
# Recommended wiring for the current FPGA top-level:
#   Raspberry Pi SCLK  -> FPGA GPIO_1_D27
#   Raspberry Pi CE0_N -> FPGA GPIO_1_D29
#   Raspberry Pi MISO  -> FPGA GPIO_1_D31
#   Raspberry Pi GPIO  -> FPGA GPIO_1_D25 (window_ready, optional but recommended)

SPI_DEVICE="${SPI_DEVICE:-/dev/spidev0.0}"
WINDOW_READY_LINE="${WINDOW_READY_LINE:-}"
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
  if ! grep -Fqx "${line}" "${CONFIG_FILE}"; then
    echo "${line}" >> "${CONFIG_FILE}"
    echo "Added: ${line}"
  else
    echo "Already present: ${line}"
  fi
}

ensure_line "dtparam=spi=on"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-pip python3-venv python3-libgpiod gpiod python3-spidev

if [[ ! -d "${PROJECT_DIR}/.venv" ]]; then
  python3 -m venv --system-site-packages "${PROJECT_DIR}/.venv"
fi

"${PROJECT_DIR}/.venv/bin/pip" install --upgrade pip
"${PROJECT_DIR}/.venv/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"

if ! "${PROJECT_DIR}/.venv/bin/python" -c "import gpiod, spidev" >/dev/null 2>&1; then
  cat <<'EOF'

Warning:
The GPIO or SPI Python modules are not visible inside .venv.
If needed, recreate .venv with --system-site-packages or install the missing package manually.
EOF
fi

cat <<EOF

Setup completed.

Next steps:
1. Reboot the Raspberry Pi:
   sudo reboot
2. After reboot, confirm SPI is enabled:
   ls -l /dev/spidev*
3. Recommended event-comparison flow:
   cd ${PROJECT_DIR}
   .venv/bin/python analyzer_from_fpga_fft.py \\
     -D ${SPI_DEVICE} \\
     --spi-max-speed-hz 8000000 \\
     --spi-mode 0 \\
     -r 48000 \\
     --frame-bins 512 \\
     --useful-bins 256 \\
     --bfpexp-hold-frames 1$( [[ -n "${WINDOW_READY_LINE}" ]] && printf ' \\\n     --window-ready-line %s' "${WINDOW_READY_LINE}" )
4. Optional second terminal for FFT visualization:
   cd ${PROJECT_DIR}
   .venv/bin/python plotFFT.py --rate 48000 --frame-bins 512
5. Optional third terminal for raw CSV logging:
   cd ${PROJECT_DIR}
   .venv/bin/python fft_spi_logger.py -D ${SPI_DEVICE} --spi-max-speed-hz 8000000 --spi-mode 0 --frame-bins 512 --bfpexp-hold-frames 1 --csv fft_capture.csv

If you wired window_ready, pass --window-ready-line ${WINDOW_READY_LINE:-<gpio-line>} to reduce idle polling.
EOF
