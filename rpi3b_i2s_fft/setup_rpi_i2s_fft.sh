#!/usr/bin/env bash
set -euo pipefail

# Configure Raspberry Pi OS for the official ACES FPGA -> Raspberry Pi I2S
# capture path and install runtime dependencies.
# Usage:
#   sudo ./setup_rpi_i2s_fft.sh
#
# Default Raspberry Pi PCM/I2S GPIO mapping used by overlays:
#   GPIO18 -> PCM_CLK  (I2S BCLK)
#   GPIO19 -> PCM_FS   (I2S LRCLK/WS)
#   GPIO20 -> PCM_DIN  (I2S SD input)
#   GPIO21 -> PCM_DOUT (I2S SD output, optional for capture-only)
#
# These defaults come from the SoC PCM/I2S peripheral pinmux (ALT functions)
# selected by the device-tree overlay, not from explicit per-pin commands here.

HOST_RATE_HZ=48828
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASOC_INSTALL_SCRIPT="${PROJECT_DIR}/asoc/install_fpgafft_overlay.sh"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo."
  exit 1
fi

if [[ ! -x "${ASOC_INSTALL_SCRIPT}" ]]; then
  echo "Official overlay installer not found or not executable: ${ASOC_INSTALL_SCRIPT}"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  build-essential \
  device-tree-compiler \
  libasound2-dev \
  raspberrypi-kernel-headers \
  python3 \
  python3-pip \
  python3-venv \
  python3-libgpiod \
  gpiod \
  alsa-utils

"${ASOC_INSTALL_SCRIPT}"

if [[ ! -d "${PROJECT_DIR}/.venv" ]]; then
  python3 -m venv --system-site-packages "${PROJECT_DIR}/.venv"
fi

"${PROJECT_DIR}/.venv/bin/pip" install --upgrade pip
"${PROJECT_DIR}/.venv/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"

gcc -O2 -Wall -Wextra -pthread \
  -o "${PROJECT_DIR}/alsa_logger" \
  "${PROJECT_DIR}/alsa_logger.c" \
  -lasound

if ! "${PROJECT_DIR}/.venv/bin/python" -c "import gpiod" >/dev/null 2>&1; then
  cat <<'EOF'

Warning:
The GPIO Python module is not visible inside .venv.
If you need BFPEXP/DONE handshake support, recreate .venv with --system-site-packages
or install gpiod inside the virtualenv manually.
EOF
fi

cat <<EOF

Setup completed.

Next steps:
1. Reboot the Raspberry Pi:
   sudo reboot
2. After reboot, list ALSA devices:
   arecord -l
3. Recommended event-comparison flow:
   cd ${PROJECT_DIR}
   .venv/bin/python analyzer_from_fpga_fft.py -r ${HOST_RATE_HZ} --frame-bins 512 --useful-bins 256 --capture-backend alsa-c
4. Optional second terminal for FFT visualization:
   cd ${PROJECT_DIR}
   .venv/bin/python plotFFT.py --rate ${HOST_RATE_HZ} --frame-bins 512
5. Optional third terminal for raw CSV logging:
   cd ${PROJECT_DIR}
   .venv/bin/python fft_i2s_logger.py -r ${HOST_RATE_HZ} --csv fft_capture.csv --capture-backend alsa-c

If auto-detection chooses the wrong input, set AUDIO_DEVICE or pass -D hw:X,Y explicitly.
The official overlay installed above keeps FPGA as I2S master and Pi as slave/capture side.
EOF
