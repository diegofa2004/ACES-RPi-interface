# Raspberry Pi FFT Bridge

This package reads tagged FFT windows from the FPGA over SPI and keeps the same host-side outputs used before: `fft.npy`, `evento.npy`, debug JSONL logs, and CSV/raw captures.

## Files

- `spi_stream.py`: SPI device discovery, open/close, and raw transfers
- `fpga_fft_adapter.py`: tagged-word receiver and FFT/MFCC extraction
- `analyzer_from_fpga_fft.py`: live analyzer, event recording, raw capture, and replay
- `fft_spi_logger.py`: raw tagged-pair CSV logger
- `setup_rpi_spi_fft.sh`: Raspberry Pi setup helper for SPI mode
- `run_channel_debug_matrix.sh`: raw capture + replay matrix for protocol debug

## Wiring

Recommended wiring for `rtl/top/top_level_test.sv`:

- Raspberry Pi `SCLK` -> FPGA `GPIO_1_D27`
- Raspberry Pi `CE0_N` -> FPGA `GPIO_1_D29`
- Raspberry Pi `MISO` -> FPGA `GPIO_1_D31`
- Raspberry Pi GPIO input -> FPGA `GPIO_1_D25` (`window_ready`, optional but recommended)

The FPGA is the SPI slave. The Raspberry Pi is the SPI master.

## Setup

```bash
cd submodules/ACES-RPi-interface/rpi3b_spi_fft
sudo ./setup_rpi_spi_fft.sh
```

After reboot:

```bash
ls -l /dev/spidev*
```

## Live Analyzer

```bash
cd submodules/ACES-RPi-interface/rpi3b_spi_fft
.venv/bin/python analyzer_from_fpga_fft.py \
  -D /dev/spidev0.0 \
  --spi-max-speed-hz 8000000 \
  --spi-mode 0 \
  --window-ready-line 23 \
  -r 48000 \
  --frame-bins 512 \
  --useful-bins 256 \
  --bfpexp-hold-frames 1
```

Important options:

- `-D, --device`: SPI device, for example `/dev/spidev0.0`
- `--spi-max-speed-hz`: Pi master clock
- `--spi-mode`: FPGA adapter expects mode `0`
- `--window-ready-line`: GPIO line used to wait for a full FFT window
- `--bfpexp-hold-frames`: must match the FPGA transport setting

## Raw CSV Logging

```bash
.venv/bin/python fft_spi_logger.py \
  -D /dev/spidev0.0 \
  --spi-max-speed-hz 8000000 \
  --spi-mode 0 \
  --window-ready-line 23 \
  --frame-bins 512 \
  --bfpexp-hold-frames 1 \
  --csv fft_capture.csv
```

## Channel Debug Matrix

```bash
./run_channel_debug_matrix.sh \
  --device /dev/spidev0.0 \
  --spi-max-speed-hz 8000000 \
  --spi-mode 0 \
  --window-ready-line 23 \
  --frame-bins 512 \
  --useful-bins 256
```

This generates:

- one shared raw capture
- one raw capture index JSONL
- one replay log per scenario
- a replay command script
- a TSV summary

## Output Contract

The SPI transaction preserves the same 32-bit tagged words already consumed by the old host flow:

- tag `1`: BFPEXP metadata
- tag `2`: FFT `(real, imag)` bin pair
- payload width: `18` signed bits
- byte order: little-endian per 32-bit word

Each SPI transaction is:

```text
BFPEXP x bfpexp_hold_frames
then
frame_bins FFT pairs
```

## Tests

From the repo root:

```bash
python3 -m unittest discover -s submodules/ACES-RPi-interface/tests -p 'test_*.py'
```
