# FPGA FFT ASoC Overlay

Esta pasta contém a infraestrutura mínima de kernel/Device Tree para o caminho
FPGA -> Raspberry Pi 3B definido em `docs/overlay_implemenation_plan.md`.

## Artefatos

- `snd-soc-fpgafft-codec.c`: codec ASoC mínimo, capture-only, sem plano de controle.
- `fpga-i2s-rx-32x2-slave-overlay.dts`: overlay `simple-audio-card` para Pi slave e FPGA master.
- `fpga-i2s-rx-32x2-slave-bitclock-inv-overlay.dts`: variante experimental com inversão de BCLK.
- `fpga-i2s-rx-32x2-slave-frame-inv-overlay.dts`: variante experimental com inversão de frame sync.
- `fpga-i2s-rx-32x2-slave-bitclock-frame-inv-overlay.dts`: variante experimental com ambas as inversões.
- `fpga-leftj-rx-32x2-slave-overlay.dts`: variante experimental em `LEFT_J` para testar `data_delay = 0`.
- `fpga-leftj-rx-32x2-slave-bitclock-inv-overlay.dts`: `LEFT_J` com inversão de BCLK.
- `fpga-leftj-rx-32x2-slave-frame-inv-overlay.dts`: `LEFT_J` com inversão de frame sync.
- `fpga-leftj-rx-32x2-slave-bitclock-frame-inv-overlay.dts`: `LEFT_J` com ambas as inversões.
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
make overlay-variants
```

## Matriz de investigação de polaridade

O codec stub agora aceita e registra no `dmesg` as quatro combinações de
inversão do `DAIFMT`. Isso permite testar se o problema está na interpretação
de polaridade pelo `bcm2835-i2s`, mesmo quando o barramento parece correto no
analisador lógico.

Overlays disponíveis:

- `fpga-i2s-rx-32x2-slave`: base atual, equivalente a `NB_NF`
- `fpga-i2s-rx-32x2-slave-bitclock-inv`: tentativa `IB_NF`
- `fpga-i2s-rx-32x2-slave-frame-inv`: tentativa `NB_IF`
- `fpga-i2s-rx-32x2-slave-bitclock-frame-inv`: tentativa `IB_IF`
- `fpga-leftj-rx-32x2-slave`: tentativa `LEFT_J + NB_NF`
- `fpga-leftj-rx-32x2-slave-bitclock-inv`: tentativa `LEFT_J + IB_NF`
- `fpga-leftj-rx-32x2-slave-frame-inv`: tentativa `LEFT_J + NB_IF`
- `fpga-leftj-rx-32x2-slave-bitclock-frame-inv`: tentativa `LEFT_J + IB_IF`

Exemplo de build manual de uma variante:

```bash
cd submodules/ACES-RPi-interface/rpi3b_i2s_fft/asoc
make fpga-i2s-rx-32x2-slave-frame-inv.dtbo
```

Checklist após instalar uma variante:

```bash
dmesg | grep -i fpgafft
dmesg | grep -i 'I2S SYNC error'
arecord --dump-hw-params -D hw:2,0 -f S32_LE -c 2 -r 48828 -d 1 /dev/null
```

A mensagem `set_fmt raw=... format=... inversion=... master=...` no `dmesg`
serve para confirmar qual polaridade o `simple-audio-card` realmente entregou
ao codec/CPU DAI.

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
