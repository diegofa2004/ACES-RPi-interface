# RPi 3B I2S Microphone Mirror

This folder supports the current ACES host flow:

- the FPGA receives microphone audio over I2S
- the FPGA keeps its own FFT only for board-side HEX/debug
- the FPGA mirrors the raw microphone I2S stream to the Raspberry Pi
- the Raspberry Pi computes FFT, MFCC, and event comparison in software

## Files

- `setup_rpi_i2s_fft.sh`: installs the overlay, ALSA tools, Python dependencies, and the native capture helper.
- `fpga_fft_adapter.py`: shared adapter that reads raw `S32_LE` stereo frames and computes FFT + MFCC locally.
- `analyzer_from_fpga_fft.py`: main detection flow with the same reference/comparison behavior already used by the project.
- `fft_i2s_logger.py`: saves raw mirrored samples to CSV and/or `.raw`.
- `verify_transport_stream.py`: checks a raw capture and reports channel dominance, correlation, and FFT peaks.
- `live_spectrogram.py`: live spectrum/spectrogram viewer driven by the mirrored stream.
- `plotFFT.py`: viewer for `fft.npy` saved by the analyzer.
- `analog_discovery_i2s_capture.py`: passive logic-capture decoder for the raw mirrored I2S bus, with optional `.raw` stereo export.
- `i2s_stream.py`: shared ALSA capture helpers.
- `spectral_features.py`: mel filter and DCT helpers used for MFCC generation.

## Wiring

Recommended topology: FPGA as I2S master, Raspberry Pi as I2S slave/capture side.

- FPGA `sck` -> RPi GPIO18 / pin 12
- FPGA `ws` -> RPi GPIO19 / pin 35
- FPGA `sd` -> RPi GPIO20 / pin 38
- GND -> GND

Notes:

- both sides must use 3.3 V logic
- keep a common ground
- if any FPGA pin is 5 V, use a level shifter before the Raspberry Pi

## Setup

```bash
cd rpi3b_i2s_fft
chmod +x setup_rpi_i2s_fft.sh
sudo ./setup_rpi_i2s_fft.sh
sudo reboot
```

After reboot:

```bash
arecord -l
```

The project auto-detects the ALSA capture device when possible. If auto-detection picks the wrong interface, pass `-D hw:X,Y` or set `AUDIO_DEVICE`.

## Main Run

The main comparison flow is:

```bash
cd rpi3b_i2s_fft
.venv/bin/python analyzer_from_fpga_fft.py -r 48828 --frame-bins 512 --useful-bins 256 --capture-backend alsa-c
```

What it does:

1. reads the raw mirrored microphone stream from ALSA
2. selects the active channel
3. computes the FFT in software
4. computes MFCC from that FFT
5. feeds the same reference/comparison path already used by `compararEvento`

Press `Enter` once to save a reference event into `evento.npy` and `fft.npy`.

## Visualization

Live spectrogram from the mirrored stream:

```bash
cd rpi3b_i2s_fft
.venv/bin/python live_spectrogram.py -r 48828 --frame-bins 512 --useful-bins 256 \
    --capture-backend alsa-c --history-seconds 12 --smooth-frames 4 --fps 12
```

Viewer for the saved `fft.npy`:

```bash
cd rpi3b_i2s_fft
.venv/bin/python plotFFT.py --spectrogram --rate 48828 --frame-bins 512
```

## Logging And Verification

Raw capture logger:

```bash
cd rpi3b_i2s_fft
.venv/bin/python fft_i2s_logger.py -r 48828 --capture-backend alsa-c \
    --csv mirror_capture.csv --raw-out mirror_capture.raw --channel-mode auto
```

Transport verifier over a previously saved raw capture:

```bash
cd rpi3b_i2s_fft
.venv/bin/python verify_transport_stream.py --raw mirror_capture.raw \
    --rate 48828 --frame-bins 512 --useful-bins 256 --channel-mode auto --json-out mirror_capture.json
```

Debug matrix helper:

```bash
cd rpi3b_i2s_fft
./run_channel_debug_matrix.sh --seconds 8 --capture-backend alsa-c
```

It captures one shared raw dump, replays that same dump through multiple channel modes, and writes:

- `mirror_capture.raw`
- `mirror_capture_auto.csv`
- one JSON summary per scenario
- one log per scenario
- `scenario_summary.tsv`
- `replay_commands.sh`

## Analog Discovery Utility

For board-level debug without the Raspberry Pi ALSA path:

```bash
cd rpi3b_i2s_fft
.venv/bin/python analog_discovery_i2s_capture.py --capture-seconds 0.01 --sample-shift-bits 8
```

This utility decodes the raw mirrored I2S words directly from `SCK/WS/SD`, exports:

- `_samples.csv`: logic-level capture timeline
- `_words.csv`: decoded raw 32-bit words
- `_frames.csv`: paired stereo frames
- `_frames.raw`: stereo `S32_LE` output compatible with `verify_transport_stream.py`

## Use From Python

```python
from rpi3b_i2s_fft import FFTAdapterConfig, FPGAFFTReceiver

cfg = FFTAdapterConfig(sample_rate=48828, frame_bins=512, useful_bins=256)
rx = FPGAFFTReceiver(cfg)
rx.start()
try:
    frame = rx.read_frame()
    if frame is not None:
        fft_bins, mfcc = frame
finally:
    rx.stop()
```

## Stream Contract

- ALSA format: `S32_LE`
- channels: `2`
- the FPGA output to the Raspberry Pi is a direct mirror of the microphone I2S stream
- the Raspberry Pi software decides which channel is useful and performs FFT/MFCC locally

## Tests

From the submodule root:

```bash
python3 -m unittest discover -s tests
```

The offline tests cover the software FFT/MFCC path, logging utilities, verification helpers, plotting helpers, and comparison support code. Hardware-dependent behavior still needs validation on the real Raspberry Pi and FPGA.
