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
- `i2s_stream.py`: shared helpers for `arecord` startup, exact reads, and clean shutdown.
- `spectral_features.py`: lightweight mel-filter and DCT helpers used for MFCC generation.

## Wiring (FPGA as I2S master - recommended for this project)

- FPGA I2S `sck` output -> RPi GPIO18 (pin 12) `BCLK`
- FPGA I2S `ws` output -> RPi GPIO19 (pin 35) `LRCLK/WS`
- FPGA I2S `sd` output -> RPi GPIO20 (pin 38) `DIN`
- GND <-> GND

Alternative (RPi as I2S master):

- RPi GPIO18 (pin 12) `BCLK` -> FPGA I2S `sck` input
- RPi GPIO19 (pin 35) `LRCLK/WS` -> FPGA I2S `ws` input
- FPGA I2S `sd` output -> RPi GPIO20 (pin 38) `DIN`

Optional GPIO handshake wires (if used):

- FPGA BFPEXP flag output -> RPi GPIO23 (pin 16) input
- RPi GPIO24 (pin 18) output -> FPGA DONE input

Backup GPIO options (if 23/24 are unavailable):

- GPIO25 (pin 22)
- GPIO16 (pin 36)
- GPIO26 (pin 37)

Electrical notes:

- FPGA and RPi GPIO must both be 3.3V logic.
- Keep a common ground between FPGA and RPi.
- If FPGA GPIO is 5V, use a level shifter before connecting to RPi GPIO.

## Setup on Raspberry Pi

For FPGA-master clocking, the Pi must be configured with an I2S/ALSA overlay that matches your hardware and supports external BCLK/LRCLK input.
The setup script uses `I2S_OVERLAY` from environment (default is `googlevoicehat-soundcard`).
If that default does not match your FPGA-master wiring, run setup with your own overlay value.

```bash
cd rpi3b_i2s_fft
chmod +x setup_rpi_i2s_fft.sh
sudo ./setup_rpi_i2s_fft.sh
```

The setup script creates `.venv` with `--system-site-packages`, so the `python3-gpiod`
package installed by `apt` is also visible inside the project virtualenv.

Reboot after setup:

```bash
sudo reboot
```

How software uses GPIO18/GPIO19:

- Python code does not bit-bang these pins.
- Device-tree overlay enables the SoC I2S peripheral and maps GPIO18/19/20 to ALT functions.
- ALSA `arecord` reads from the configured I2S capture device (`hw:2,0`).
- In FPGA-master mode, FPGA drives BCLK and LRCLK; the Pi I2S peripheral samples data on GPIO20 using those clocks.

Where these default pins come from:

- Raspberry Pi SoC exposes the PCM/I2S peripheral on a standard pinmux mapping.
- Overlays select that peripheral function, so GPIO18/19/20/21 become PCM_CLK/PCM_FS/PCM_DIN/PCM_DOUT.
- This project uses GPIO18 (BCLK), GPIO19 (LRCLK), and GPIO20 (DIN) for capture.

How to verify pin function and clock direction on-device:

1. Confirm pinmux function (should show PCM/ALT function):

```bash
pinctrl get 18
pinctrl get 19
pinctrl get 20
pinctrl get 21
```

2. Confirm overlay/device was loaded:

```bash
aplay -l
arecord -l
```

3. Confirm clocks are actually present from FPGA (FPGA-master mode):

- Use a scope/logic analyzer on GPIO18 (BCLK) and GPIO19 (LRCLK).
- In FPGA-master mode, these clocks must be driven by FPGA while capture is active.
- If clocks are missing, Pi cannot capture regardless of Python settings.

Notes:

- Linux pinmux reports function selection, not electrical "input/output" direction in the same way as regular GPIO.
- In I2S mode, the peripheral role (master/slave) and external hardware determine who drives clocks.

## Run

Start daemon (adjust `-D` after checking `arecord -l`):

```bash
cd rpi3b_i2s_fft
.venv/bin/python fft_i2s_daemon.py -D hw:2,0 -r 48000
```

In another shell, read latest values:

```bash
cd rpi3b_i2s_fft
.venv/bin/python fft_i2s_client.py --watch
```

Log full stream while still updating shared memory:

```bash
cd rpi3b_i2s_fft
.venv/bin/python fft_i2s_logger.py -D hw:2,0 -r 48000 --csv fft_capture.csv
```

## Use from another Python program

If you import from outside this folder, add `submodules/ACES-RPi-interface` to `PYTHONPATH`
or otherwise make the package parent directory visible to Python.

```python
from rpi3b_i2s_fft.fft_shared import FFTSharedState

state = FFTSharedState(name="fft_i2s_latest", create=False)
item = state.read()
real = item["real"]
imag = item["imag"]
# pass real/imag to your analyzer
state.close()
```

## Stream format expected from FPGA

- Common transport: `S32_LE`, 2 channels
- Left channel = real, right channel = imag (unless you intentionally swap)

Raw mode (no in-band tags):

- Use full 32-bit signed values directly.
- If your source data is 18-bit, sign-extend to 32-bit before transmit.

Tagged mode (with in-band BFPEXP/FFT/idle tags):

- Upper bits carry tag metadata, so payload is not full 32-bit signed anymore.
- Default payload is signed 18-bit (documented below in the tagged section).

## Migrating your friend's analyzer from serial audio to FPGA FFT bins

Your old pipeline was:

- bytes -> int16 samples
- `calculaFFT(samples)`
- `calculaMFCC(fftBins, mel_filter)`

With FPGA FFT over I2S, use this pipeline instead:

- I2S frame -> complex bins `(real, imag)`
- magnitude bins `sqrt(real^2 + imag^2)`
- MFCC from magnitude bins

MFCC generation is implemented locally with `numpy`, so this project no longer depends
on `librosa` or `scipy` just to build the mel filterbank and DCT basis.

Run the adapter example:

```bash
cd rpi3b_i2s_fft
.venv/bin/python analyzer_from_fpga_fft.py -D hw:2,0 -r 48000 --frame-bins 512 --useful-bins 256
```

GPIO handshake mode (optional):

- `--bfpexp-flag-line`: input line that is active while FPGA sends BFPEXP.
- `--done-line`: output line pulsed by RPi after exactly 512 bins are consumed.
- default trigger is BFPEXP falling edge (high -> low), matching "BFPEXP then FFT".

Dependency note for handshake mode:

- GPIO mode requires Python `gpiod` module support at runtime.
- The setup script already installs `python3-gpiod` and exposes it inside `.venv`.
- If you use a custom virtualenv and import still fails, install it inside that env:

```bash
cd rpi3b_i2s_fft
.venv/bin/pip install gpiod
```

Example (line numbers are GPIO chip offsets):

```bash
.venv/bin/python analyzer_from_fpga_fft.py -D hw:2,0 -r 48000 \
	--frame-bins 512 --useful-bins 256 \
	--bfpexp-flag-line 23 --done-line 24
```

Notes about framing reliability:

- GPIO-only framing works, but ALSA buffering means trigger timing and sample boundaries are not perfectly phase-locked.
- For best robustness, keep streaming all the time and include in-band framing (start marker or bin index + valid bit).
- Zero-fill by itself is not enough to mark frame boundaries because true FFT bins can also be zero.

In-band tagged stream mode (BFPEXP + FFT + idle over I2S):

- Enable with `--use-i2s-tags`.
- Each 32-bit I2S word carries a small type tag plus signed payload bits.
- Default mapping used by the receiver:
	- `tag_shift=30`, `tag_mask=0x3` (2 tag bits in bits 31..30)
	- `payload_bits=18` (signed payload in bits 17..0)
	- tags: `0=idle`, `1=BFPEXP`, `2=FFT`

Frame start logic in tagged mode:

- Receiver waits for BFPEXP-tagged words.
- After BFPEXP is seen, the first FFT-tagged pair starts the FFT frame.
- Receiver then counts exactly 512 FFT-tagged complex pairs.
- On completion, DONE GPIO is pulsed (if `--done-line` is configured).

Example tagged mode run:

```bash
.venv/bin/python analyzer_from_fpga_fft.py -D hw:2,0 -r 48000 \
	--frame-bins 512 --useful-bins 256 \
	--use-i2s-tags --tag-shift 30 --tag-mask 0x3 --payload-bits 18 \
	--tag-idle 0 --tag-bfpexp 1 --tag-fft 2 \
	--done-line 24
```

If your FPGA cannot guarantee BFPEXP tags before FFT tags, add:

```bash
--allow-fft-without-bfpexp
```

FPGA transmit reference (example RTL behavior):

- Keep I2S running continuously.
- Send tagged words for every LRCLK slot.
- Recommended phase order per cycle:
	1. `BFPEXP`: tag=`1`
	2. `FFT`: tag=`2` for 512 complex pairs
	3. `IDLE`: tag=`0` until next cycle

Suggested 32-bit packing (matches default Python decoder):

- bits `[31:30]` = `tag`
- bits `[29:18]` = reserved (`0`)
- bits `[17:0]` = signed payload (2's complement)

Verilog-style helper:

```verilog
function automatic [31:0] pack_word;
	input [1:0]  tag;
	input signed [17:0] payload;
	begin
		pack_word = {tag, 12'd0, payload[17:0]};
	end
endfunction
```

Per-I2S-frame mapping example:

```verilog
// LRCLK left slot then right slot
// Use the same tag on both channels for a given semantic type.
left_word  <= pack_word(tag_kind, left_payload);
right_word <= pack_word(tag_kind, right_payload);
```

State-machine sketch:

```verilog
case (state)
	ST_BFPEXP: begin
		tag_kind <= 2'd1;
		// emit BFPEXP payload stream
		if (bfpexp_last) state <= ST_FFT;
	end

	ST_FFT: begin
		tag_kind <= 2'd2;
		// emit FFT bins as complex pairs
		// left/right order must match the RPi decode assumption
		if (fft_bin_idx == 9'd511 && sample_accepted) state <= ST_WAIT_DONE;
	end

	ST_WAIT_DONE: begin
		tag_kind <= 2'd0; // idle while waiting
		if (done_from_rpi_sync) state <= ST_BFPEXP;
	end
endcase
```

Important hardware notes:

- Synchronize `done_from_rpi` into FPGA clock domain with 2 flip-flops.
- If possible, hold each I2S word stable until shifted out (no combinational change mid-word).
- If your left/right are swapped (imag/real), adjust either FPGA mapping or Python decoding consistently.

Integration note:

- Keep your `compararEvento(buffer2, buffer4, lock, get_lastEventTime)` logic unchanged.
- Replace only the serial-read loop with calls to `FPGAFFTReceiver.read_frame()` and append:
	- `buffer2.append(mfcc[:8])`
	- `buffer4.append(fft_bins)`
