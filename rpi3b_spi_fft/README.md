# Raspberry Pi I2S Audio Bridge

This package now receives raw microphone audio from the FPGA over I2S/ALSA.
The Raspberry Pi computes the FFT and MFCC locally, then keeps the same
`fft.npy`, `evento.npy`, trigger flow, and comparison logic used before.

## Current data path

```text
microphone -> FPGA clocking + raw I2S pass-through -> Raspberry Pi ALSA
-> FFT on Raspberry Pi -> MFCC on Raspberry Pi -> event comparison
```

## Relevant files

- `i2s_stream.py`: ALSA device discovery and capture command helpers
- `fpga_audio_adapter.py`: PCM reader plus FFT/MFCC extraction on the Pi
- `analyzer_from_fpga_fft.py`: live analyzer, event recording, and comparison
- `live_spectrogram.py`: rolling FFT viewer driven by the same raw audio stream
- `alsa_logger.c`: native ALSA capture helper for low-overhead raw mode
- `asoc/`: Raspberry Pi overlay and codec stub for FPGA -> Pi I2S capture

## FPGA wiring to Raspberry Pi

In `rtl/top/top_level_test.sv` the raw I2S link to the Raspberry Pi is mirrored
on dedicated header pins:

- FPGA `GPIO_1_D17` -> Raspberry Pi `GPIO18 / PCM_CLK`
- FPGA `GPIO_1_D19` -> Raspberry Pi `GPIO19 / PCM_FS`
- FPGA `GPIO_1_D20` -> Raspberry Pi `GPIO20 / PCM_DIN`

The FPGA remains master of `BCLK` and `LRCLK`. The Raspberry Pi stays in
capture/slave mode through the ASoC overlay.

## Setup on Raspberry Pi

Build/install the overlay from:

```bash
cd submodules/ACES-RPi-interface/rpi3b_spi_fft/asoc
sudo ./install_fpgafft_overlay.sh
```

Then confirm the ALSA device exists:

```bash
arecord -l
arecord --dump-hw-params -D hw:2,0 -f S32_LE -c 2 -r 48828 -d 1 /dev/null
```

## Live analyzer

```bash
cd submodules/ACES-RPi-interface/rpi3b_spi_fft
.venv/bin/python analyzer_from_fpga_fft.py \
  -D hw:2,0 \
  --capture-backend auto \
  -r 48828 \
  --frame-bins 512 \
  --useful-bins 256 \
  --read-frames 512 \
  --sample-shift-bits 8 \
  --mono-channel auto
```

Notes:

- `--sample-shift-bits 8` is the expected default when the 24-bit microphone
  sample is packed inside a 32-bit ALSA slot.
- `--mono-channel auto` automatically picks the active channel when only one
  I2S slot carries the microphone data.
- Press `ENTER` or touch `record_button.trigger` through the GPIO helper to
  save a reference event.

## Native capture helper

If you want to avoid `arecord`, compile the native helper:

```bash
cd submodules/ACES-RPi-interface/rpi3b_spi_fft
gcc -O2 -Wall -Wextra -pthread -o alsa_logger alsa_logger.c -lasound
```

Then run the analyzer with:

```bash
.venv/bin/python analyzer_from_fpga_fft.py \
  -D hw:2,0 \
  --capture-backend alsa-c \
  --capture-binary ./alsa_logger
```

## Tests

From the repo root:

```bash
python3 -m unittest discover \
  -s submodules/ACES-RPi-interface/tests \
  -p 'test_fpga_audio_adapter.py'

python3 -m unittest discover \
  -s submodules/ACES-RPi-interface/tests \
  -p 'test_analyzer_from_fpga_fft.py'

python3 -m unittest discover \
  -s submodules/ACES-RPi-interface/tests \
  -p 'test_comparar_evento.py'
```
