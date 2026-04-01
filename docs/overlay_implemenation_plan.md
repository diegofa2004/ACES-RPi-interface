# FPGA -> Raspberry Pi 3B ASoC / Overlay Implementation Plan

Este documento e a implementacao em `submodules/ACES-RPi-interface/rpi3b_i2s_fft/asoc/`
sao a fonte de verdade da infraestrutura minima de kernel/overlay para o projeto
FPGA -> Raspberry Pi.

## 1. Objetivo desta etapa

Entregar a infraestrutura minima, correta e auditavel para que o Raspberry Pi 3B
exponha uma sound card ALSA/ASoC de captura I2S vinda da FPGA, mantendo toda a
semantica do protocolo no user space.

O resultado esperado desta etapa e:

- sound card ALSA/ASoC estavel e identificavel;
- captura `S32_LE`, 2 canais, 32 bits por slot;
- FPGA sempre como mestre de `BCLK` e `LRCLK`;
- Raspberry Pi sempre como slave/capture side;
- raw mode e tagged mode compartilhando exatamente o mesmo transporte;
- parsing do protocolo inteiramente fora do kernel.

## 2. Premissas fechadas

Estas decisoes nao devem ser alteradas por esta camada:

1. O kernel nao interpreta BFPEXP, FFT ou IDLE.
2. O kernel so transporta words.
3. O codec do projeto e um stub ASoC minimo, sem plano de controle.
4. Nao existe configuracao da FPGA por I2C, SPI ou controle equivalente.
5. O overlay oficial usa `simple-audio-card`.
6. A FPGA e sempre `bitclock-master` e `frame-master`.
7. O enlace e sempre:
   - formato `I2S` Philips
   - 2 slots
   - 32 bits por slot
   - `S32_LE` no host
8. A taxa fisica nominal do enlace e `48 828.125 Hz`.
9. A taxa inteira exposta ao ALSA/host e `48828 Hz`.

## 3. Implementacao final

### 3.1 Localizacao dos artefatos

Todos os artefatos especificos de ASoC/overlay foram colocados em:

`submodules/ACES-RPi-interface/rpi3b_i2s_fft/asoc/`

Arquivos implementados:

- `snd-soc-fpgafft-codec.c`
- `fpga-i2s-rx-32x2-slave-overlay.dts`
- `Makefile`
- `install_fpgafft_overlay.sh`
- `README.md`

Arquivos atualizados para refletir o fluxo final:

- `submodules/ACES-RPi-interface/rpi3b_i2s_fft/setup_rpi_i2s_fft.sh`
- `submodules/ACES-RPi-interface/rpi3b_i2s_fft/README.md`
- `submodules/ACES-RPi-interface/rpi3b_i2s_fft/i2s_stream.py`
- `submodules/ACES-RPi-interface/rpi3b_i2s_fft/fpga_fft_adapter.py`
- `submodules/ACES-RPi-interface/rpi3b_i2s_fft/analyzer_from_fpga_fft.py`
- `submodules/ACES-RPi-interface/rpi3b_i2s_fft/fft_i2s_logger.py`
- `submodules/ACES-RPi-interface/rpi3b_i2s_fft/plotFFT.py`
- `submodules/ACES-RPi-interface/rpi3b_i2s_fft/alsa_logger.c`

### 3.2 Nomes oficiais

Os nomes oficiais adotados nesta implementacao sao:

- nome do modulo/driver: `snd-soc-fpgafft-codec`
- `compatible` do codec em DT: `aces,fpgafft-codec`
- nome do DAI do codec: `fpgafft-codec-dai`
- nome estavel da sound card ALSA: `aces-fpgafft`
- nome do overlay: `fpga-i2s-rx-32x2-slave`

Esses nomes foram escolhidos para serem:

- especificos do projeto;
- curtos o suficiente para `arecord -l`;
- faceis de buscar em logs e scripts;
- sem dependencia de nomes de HATs de terceiros.

## 4. Decisao tecnica para o codec minimo

### 4.1 O que o codec faz

O arquivo `snd-soc-fpgafft-codec.c` implementa um codec ASoC minimo, capture-only,
registrado como `platform_driver` casado por Device Tree.

Ele faz apenas o necessario para fechar a topologia com `simple-audio-card`:

- registra um componente ASoC;
- registra um DAI de captura;
- aceita `I2S` Philips;
- aceita `2` slots de `32` bits;
- expõe somente `S32_LE`;
- restringe a abertura do host para `48828 Hz`.

### 4.2 O que o codec nao faz

Ele nao:

- interpreta protocolo;
- conhece tags;
- configura FPGA;
- fala I2C/SPI;
- expõe mixers, ganhos ou controles;
- implementa DAPM sofisticado.

### 4.3 Justificativa

O codec existe apenas para fornecer o endpoint `sound-dai` do lado externo.
Toda a semantica do stream continua no user space, como ja estava decidido.

## 5. Decisao tecnica para o overlay

### 5.1 Topologia registrada

O overlay `fpga-i2s-rx-32x2-slave-overlay.dts` faz duas coisas:

1. habilita `&i2s`;
2. cria a sound card `simple-audio-card`.

Topologia logica:

- CPU DAI: bloco `i2s` do Raspberry Pi
- Codec DAI: stub `aces,fpgafft-codec`
- Card: `simple-audio-card`

### 5.2 Como o codec e referenciado

O overlay cria um no raiz:

- `fpgafft_codec: fpgafft-codec { compatible = "aces,fpgafft-codec"; #sound-dai-cells = <0>; }`

Depois o `simple-audio-card,codec` referencia:

- `sound-dai = <&fpgafft_codec>;`

As propriedades:

- `simple-audio-card,bitclock-master`
- `simple-audio-card,frame-master`

apontam para o subno codec-side do link (`fpgafft_codec_link`), deixando a
topologia explicitamente alinhada com FPGA master / Pi slave.

### 5.3 Parametros fixos do link

O overlay fixa:

- `simple-audio-card,format = "i2s"`
- `dai-tdm-slot-num = <2>`
- `dai-tdm-slot-width = <32>`

Isso vale tanto para raw quanto para tagged, porque o kernel nao diferencia os modos.

## 6. Taxa de amostragem: verdade de hardware vs verdade de host

### 6.1 Taxa fisica

A taxa real do enlace continua sendo:

- `48 828.125 Hz`

Essa verdade de hardware vem do clock fixo da FPGA.

### 6.2 Taxa exposta ao ALSA

O driver restringe o host para:

- `48828 Hz`

Motivo: a API ALSA trabalha com taxa inteira em Hz. Nesta camada minima nao ha
vantagem em inventar semantica especial no kernel para representar o `0.125 Hz`
residual. O contrato correto fica:

- documentar a taxa fisica nominal real;
- abrir o host em `48828`;
- validar no hardware que o transporte bruto esta coerente.

## 7. Fluxo oficial de build

Build manual no Raspberry Pi:

```bash
cd submodules/ACES-RPi-interface/rpi3b_i2s_fft/asoc
make all
```

O `Makefile` produz:

- `snd-soc-fpgafft-codec.ko`
- `fpga-i2s-rx-32x2-slave.dtbo`

Dependencias esperadas no Raspberry Pi:

- `build-essential`
- `device-tree-compiler`
- `raspberrypi-kernel-headers`

## 8. Fluxo oficial de instalacao

Instalacao direta:

```bash
cd submodules/ACES-RPi-interface/rpi3b_i2s_fft/asoc
sudo ./install_fpgafft_overlay.sh
```

O instalador oficial faz:

- localiza `config.txt` em `/boot/firmware/config.txt` ou `/boot/config.txt`;
- cria backup versionado do `config.txt`;
- recompila modulo e overlay, salvo `--skip-build`;
- instala o `.dtbo` no diretorio correto de overlays;
- instala o `.ko` em `/lib/modules/$(uname -r)/extra/`;
- executa `depmod -a`;
- garante `dtoverlay=fpga-i2s-rx-32x2-slave`;
- opcionalmente garante `dtparam=i2s=on`.

Wrapper mais completo para preparar tambem Python e `.venv`:

```bash
cd submodules/ACES-RPi-interface/rpi3b_i2s_fft
sudo ./setup_rpi_i2s_fft.sh
```

Esse wrapper instala dependencias de build/runtime e delega a instalacao do
overlay ao script oficial em `asoc/`.

## 9. Criterios de validacao pos-reboot

Depois de instalar, reinicie:

```bash
sudo reboot
```

Checklist obrigatorio:

```bash
arecord -l
aplay -l
pinctrl get 18
pinctrl get 19
pinctrl get 20
pinctrl get 21
dmesg -l err,warn
arecord --dump-hw-params -D hw:X,Y
```

Resultados esperados:

- aparece uma card identificavel como `aces-fpgafft`;
- o pinmux coloca GPIO18/19/20/21 em funcao PCM/I2S;
- nao ha erro relevante de overlay/ASoC no boot;
- o device aceita captura `S32_LE`, 2 canais.

## 10. Validacao obrigatoria do conteudo bruto

Antes de voltar ao parser Python, validar em hexadecimal:

```bash
cd submodules/ACES-RPi-interface/rpi3b_i2s_fft
gcc -O2 -Wall -Wextra -o alsa_logger alsa_logger.c -lasound
./alsa_logger hw:X,Y 48828
```

Objetivo desta etapa:

- verificar words esquerdo/direito em hexadecimal;
- confirmar que o transporte bruto esta coerente;
- tirar kernel/overlay/ASoC da posicao de suspeito principal.

So depois disso retomar:

- parser Python;
- raw mode semantico;
- tagged mode semantico;
- BFPEXP/FFT/IDLE no user space.

## 11. Limitacoes conhecidas

1. O driver nao representa `48 828.125 Hz` com fracao no ALSA; ele expoe `48828`.
2. O modulo e o overlay foram implementados para build no Raspberry Pi alvo; este
   workspace nao tem kernel headers do Pi nem hardware para validacao final de boot.
3. O sucesso de `arecord -l` nao prova que o stream semantico esta correto.
4. Se o stream bruto estiver errado, a causa ainda pode estar no RTL I2S da FPGA.

## 12. Aceitacao desta etapa

Esta etapa e considerada entregue quando:

1. o modulo `snd-soc-fpgafft-codec` compila e instala;
2. o overlay `fpga-i2s-rx-32x2-slave` instala sem editar `config.txt` manualmente;
3. a sound card `aces-fpgafft` aparece em `arecord -l`;
4. o device abre como `S32_LE`, 2 canais, `48828 Hz`;
5. o utilitario C mostra words hex coerentes;
6. raw e tagged continuam compartilhando o mesmo transporte bruto;
7. o kernel permanece semanticamente neutro em relacao ao protocolo.
