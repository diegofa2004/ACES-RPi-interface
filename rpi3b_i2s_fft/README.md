# RPi 3B I2S FFT Bridge

This folder configures Raspberry Pi OS for I2S capture and publishes latest FFT values (real, imag) through shared memory, so other programs can read them as variables.

## Files

- `setup_rpi_i2s_fft.sh`: enables I2S in boot config and installs dependencies.
- `fft_i2s_daemon.py`: runs `arecord`, decodes stereo int32 stream, writes latest values to shared memory.
- `fft_i2s_logger.py`: same capture path as daemon, but also logs every sample pair to CSV.
- `fft_i2s_client.py`: reads shared memory once or continuously.
- `fft_shared.py`: shared-memory API used by other Python programs.
- `fpga_fft_adapter.py`: converts FPGA complex FFT bins from I2S into magnitude bins + MFCC.
- `analyzer_from_fpga_fft.py`: example loop that fills analyzer buffers from FPGA FFT stream.

## Wiring (RPi as I2S master recommended)

- RPi GPIO18 (pin 12) `BCLK` -> FPGA I2S `sck` input
- RPi GPIO19 (pin 35) `LRCLK/WS` -> FPGA I2S `ws` input
- FPGA I2S `sd` output -> RPi GPIO20 (pin 38) `DIN`
- GND <-> GND

## Setup on Raspberry Pi

```bash
cd rpi3b_i2s_fft
chmod +x setup_rpi_i2s_fft.sh
sudo ./setup_rpi_i2s_fft.sh
```

Reboot after setup:

```bash
sudo reboot
```

## Run

Start daemon (adjust `-D` after checking `arecord -l`):

```bash
cd rpi3b_i2s_fft
.venv/bin/python fft_i2s_daemon.py -D hw:0,0 -r 48000
```

In another shell, read latest values:

```bash
cd rpi3b_i2s_fft
.venv/bin/python fft_i2s_client.py --watch
```

Log full stream while still updating shared memory:

```bash
cd rpi3b_i2s_fft
.venv/bin/python fft_i2s_logger.py -D hw:0,0 -r 48000 --csv fft_capture.csv
```

## Use from another Python program

```python
from fft_shared import FFTSharedState

state = FFTSharedState(name="fft_i2s_latest", create=False)
item = state.read()
real = item["real"]
imag = item["imag"]
# pass real/imag to your analyzer
state.close()
```

## Stream format expected from FPGA

- `S32_LE`, 2 channels
- Left channel = real
- Right channel = imag
- 18-bit FFT values should be sign-extended in FPGA to 32-bit

## Migrating your friend's analyzer from serial audio to FPGA FFT bins

Your old pipeline was:

- bytes -> int16 samples
- `calculaFFT(samples)`
- `calculaMFCC(fftBins, mel_filter)`

With FPGA FFT over I2S, use this pipeline instead:

- I2S frame -> complex bins `(real, imag)`
- magnitude bins `sqrt(real^2 + imag^2)`
- MFCC from magnitude bins

Run the adapter example:

```bash
cd rpi3b_i2s_fft
.venv/bin/python analyzer_from_fpga_fft.py -D hw:0,0 -r 48000 --frame-bins 512 --useful-bins 256
```

Integration note:

- Keep your `compararEvento(buffer2, buffer4, lock, get_lastEventTime)` logic unchanged.
- Replace only the serial-read loop with calls to `FPGAFFTReceiver.read_frame()` and append:
	- `buffer2.append(mfcc[:8])`
	- `buffer4.append(fft_bins)`
