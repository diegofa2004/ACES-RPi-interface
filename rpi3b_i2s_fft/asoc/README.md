# FPGA FFT ASoC Overlay

Esta pasta contém a infraestrutura mínima de kernel/Device Tree para o caminho
FPGA -> Raspberry Pi 3B definido em `docs/overlay_implemenation_plan.md`.

## Artefatos

- `snd-soc-fpgafft-codec.c`: codec ASoC mínimo, capture-only, sem plano de controle.
- `fpga-i2s-rx-32x2-slave-overlay.dts`: overlay `simple-audio-card` para Pi slave e FPGA master.
- `Makefile`: build local do módulo e do `.dtbo`.
- `install_fpgafft_overlay.sh`: instalador oficial.

## Build manual no Raspberry Pi

Pacotes esperados:

```bash
sudo apt-get install -y build-essential device-tree-compiler raspberrypi-kernel-headers
```

Compilação:

```bash
cd submodules/ACES-RPi-interface/rpi3b_i2s_fft/asoc
make all
```

## Instalação oficial

```bash
cd submodules/ACES-RPi-interface/rpi3b_i2s_fft/asoc
sudo ./install_fpgafft_overlay.sh
```

## Taxa do enlace

- Taxa física nominal do enlace: `48 828.125 Hz`
- Taxa inteira exposta ao ALSA/host: `48828`

Essa diferença de `0.125 Hz` é documentada porque o enlace físico vem de um
divisor fixo na FPGA, enquanto a API ALSA trabalha com taxa inteira em Hz.
