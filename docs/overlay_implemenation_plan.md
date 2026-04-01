A seguir está a **documentação técnica completa** para servir como fonte de verdade da implementação do overlay, do codec mínimo e do script de instalação. Ela já incorpora a decisão final de arquitetura:

* **overlay próprio**
* **codec mínimo próprio**
* **FPGA sempre como I2S master**
* **Pi sempre como slave/capture side**
* **o kernel só transporta words**
* **parsing do protocolo fica no user space**

O Raspberry Pi carrega overlays via `dtoverlay=` no `config.txt`, e o binding `simple-audio-card` é apropriado para descrever a ligação entre o DAI do SoC e um codec-side DAI com propriedades como `format`, `bitclock-master`, `frame-master`, `dai-tdm-slot-num` e `dai-tdm-slot-width`. ([Raspberry Pi][1])

---

# Documento técnico de referência

## FPGA I2S FFT Capture Sound Card para Raspberry Pi 3B

## 1. Objetivo

Criar uma solução mínima, robusta e auditável para que o Raspberry Pi 3B enxergue, via ALSA/ASoC, uma **sound card de captura I2S** conectada à FPGA.

Essa solução deve permitir:

* criação de uma sound card estável no Linux;
* captura correta dos **words de 32 bits** que chegam da FPGA;
* funcionamento tanto para **raw mode** quanto para **tagged mode**;
* independência entre a camada de transporte no kernel e a interpretação do protocolo no user space.

---

## 2. Escopo funcional

### 2.1 O que a solução deve fazer

A solução deve:

* habilitar o bloco I2S/PCM do Raspberry Pi;
* registrar uma sound card ALSA/ASoC própria do projeto;
* operar com **I2S Philips**;
* operar com **2 slots fixos de 32 bits** por frame;
* operar com a FPGA fornecendo:

  * `BCLK`
  * `LRCLK/WS`
* permitir que aplicações user space leiam os words como `S32_LE`, 2 canais.

### 2.2 O que a solução não deve fazer

A solução **não deve**:

* interpretar BFPEXP/FFT/IDLE no kernel;
* validar tags no kernel;
* reconstruir frames FFT no kernel;
* negociar ou configurar parâmetros na FPGA;
* oferecer plano de controle entre kernel e FPGA.

---

## 3. Decisão arquitetural

A arquitetura escolhida é:

### 3.1 Overlay próprio

Será criado um overlay próprio do projeto para Raspberry Pi, carregado por `dtoverlay=` no boot. O mecanismo oficial de overlays do Raspberry Pi passa pelo `config.txt`. ([Raspberry Pi][1])

### 3.2 Codec mínimo próprio

Será implementado um **codec mínimo próprio**, cuja única função é existir como endpoint `sound-dai` para o ASoC, permitindo que o `simple-audio-card` feche a topologia.

Esse codec:

* não controla a FPGA;
* não possui interface de configuração;
* não interpreta o protocolo;
* existe apenas para expor os parâmetros corretos do enlace ao ALSA.

### 3.3 User space como camada semântica

Toda a interpretação do conteúdo dos words continua fora do kernel:

* raw mode
* tagged mode
* BFPEXP/FFT/IDLE
* reconstrução de quadro FFT
* MFCC
* comparação de eventos

---

## 4. Premissas técnicas fechadas

Estas premissas são parte do contrato do sistema:

* plataforma: **Raspberry Pi 3B**
* barramento: **I2S Philips**
* direção: **captura**
* FPGA: **sempre master**
* Raspberry Pi: **sempre slave**
* slots por frame: **2**
* largura por slot: **32 bits**
* formato ALSA esperado: **`S32_LE`**
* tagged e raw compartilham o mesmo transporte
* os 32 bits do slot sempre são válidos como word completo no host
* taxa nominal:

  * `BCLK = 3,125 MHz`
  * `LRCLK = 48 828,125 Hz`
* essa taxa decorre de clock de 50 MHz na FPGA e divisão fixa conforme o projeto

O binding do `simple-audio-card` suporta exatamente os parâmetros que precisamos fixar: `format`, `bitclock-master`, `frame-master`, `dai-tdm-slot-num` e `dai-tdm-slot-width`. ([Kernel.org][2])

---

## 5. Motivação para usar codec mínimo próprio

Foi considerada a possibilidade de usar um codec pronto, mas a decisão final é usar um **codec mínimo próprio** por estes motivos:

* a FPGA não oferece plano de controle;
* os parâmetros do enlace são totalmente fixos;
* não há ganho real em reaproveitar um codec rico;
* um codec pronto pode impor restrições incompatíveis;
* o objetivo do codec aqui é apenas criar a sound card correta.

Portanto, o codec do projeto deve ser entendido como um **stub ASoC de captura digital**, e não como um codec de áudio tradicional com registradores, ganho, mixer ou configuração dinâmica.

---

## 6. Modelo em camadas

A solução fica organizada em três camadas.

### 6.1 Camada 1 — hardware físico

* FPGA transmite I2S
* Pi recebe I2S
* clocks externos vindos da FPGA

### 6.2 Camada 2 — kernel/ASoC/ALSA

* overlay habilita I2S e registra sound card
* codec mínimo próprio fornece `sound-dai`
* `simple-audio-card` conecta CPU DAI ↔ codec DAI
* ALSA expõe dispositivo de captura

### 6.3 Camada 3 — user space

* utilitário C lê words hex
* scripts Python capturam stream bruto
* parser interpreta raw/tagged
* aplicação entende BFPEXP/FFT/IDLE

---

## 7. Topologia lógica ASoC

A topologia desejada é:

* **CPU DAI**: bloco I2S/PCM do SoC do Raspberry Pi
* **Codec DAI**: codec mínimo próprio do projeto
* **Card**: `simple-audio-card`

### 7.1 Papel dos clocks

Como a FPGA é sempre master, o codec-side do enlace deve ser declarado como:

* `bitclock-master`
* `frame-master`

Isso indica, na topologia ASoC, que o lado externo é o originador dos clocks do enlace I2S. O binding do `simple-audio-card` prevê exatamente esse modelo. ([Kernel.org][2])

---

## 8. Especificação do codec mínimo

## 8.1 Objetivo do codec mínimo

O codec mínimo deve existir apenas para:

* registrar um componente ASoC;
* registrar um DAI de captura;
* declarar os parâmetros fixos do enlace;
* permitir que o `simple-audio-card` registre a card.

## 8.2 Requisitos do codec mínimo

O codec mínimo deve:

* expor **capture only**
* expor **2 canais**
* expor **32 bits por sample**
* expor taxa nominal do projeto
* não depender de I2C/SPI
* não depender de registradores programáveis
* não exigir controles DAPM complexos
* não implementar plano de controle com a FPGA

## 8.3 Requisitos negativos

O codec mínimo não deve:

* reconfigurar taxa
* trocar formato
* alterar largura de slot
* entender tags
* decidir framing FFT
* aplicar filtros ou transformações

---

## 9. Especificação do overlay

## 9.1 Objetivo do overlay

O overlay deve:

* habilitar o `&i2s`;
* criar a sound card do projeto;
* usar `compatible = "simple-audio-card"`;
* ligar o `sound-dai` do Pi ao `sound-dai` do codec mínimo;
* fixar:

  * `format = "i2s"`
  * `dai-tdm-slot-num = <2>`
  * `dai-tdm-slot-width = <32>`
  * lado externo como `bitclock-master` e `frame-master`

## 9.2 Estrutura esperada

O `.dts` do overlay deverá ter:

* `fragment@0`: habilitação do `&i2s`
* `fragment@1`: criação da sound card na raiz
* `simple-audio-card,cpu`
* `simple-audio-card,codec`

A documentação oficial do Raspberry Pi cobre o uso de overlays via `config.txt`, e o binding do `simple-audio-card` cobre a forma do card. ([Raspberry Pi][1])

## 9.3 Esqueleto conceitual

```dts
/dts-v1/;
/plugin/;

/ {
    compatible = "brcm,bcm2835";

    fragment@0 {
        target = <&i2s>;
        __overlay__ {
            status = "okay";
        };
    };

    fragment@1 {
        target-path = "/";
        __overlay__ {
            fpgafft_sound: fpgafft-sound {
                compatible = "simple-audio-card";
                simple-audio-card,name = "fpgafft";
                simple-audio-card,format = "i2s";

                simple-audio-card,bitclock-master = <&fpgafft_codec>;
                simple-audio-card,frame-master = <&fpgafft_codec>;

                simple-audio-card,cpu {
                    sound-dai = <&i2s>;
                    dai-tdm-slot-num = <2>;
                    dai-tdm-slot-width = <32>;
                };

                fpgafft_codec: simple-audio-card,codec {
                    sound-dai = <&fpgafft_codec_dai>;
                    dai-tdm-slot-num = <2>;
                    dai-tdm-slot-width = <32>;
                };
            };
        };
    };
};
```

Esse trecho é apenas conceitual; os phandles e o nó concreto do codec mínimo dependerão de como o driver será registrado.

---

## 10. Taxa de amostragem e política de abertura do ALSA

A taxa nominal do projeto é **48 828,125 Hz**.

### 10.1 Verdade de hardware

Essa taxa deve ser documentada como taxa real do enlace.

### 10.2 Verdade de software

A abertura do ALSA deve buscar essa taxa primeiro.

### 10.3 Risco conhecido

Nem todo caminho ALSA aceita qualquer taxa arbitrária de forma perfeita. Portanto, o processo de validação deve verificar:

* se o device aceita essa taxa;
* se ajusta para perto dela;
* se o comportamento continua semanticamente correto.

---

## 11. Integração com raw e tagged mode

O overlay e o codec mínimo devem ser **agnósticos ao conteúdo** do word.

Isso significa:

* raw mode: tratado como words estéreo de 32 bits
* tagged mode: tratado como words estéreo de 32 bits
* o kernel não distingue os dois

Essa neutralidade é obrigatória, porque a responsabilidade semântica foi fixada para o user space.

---

## 12. Nomeação e artefatos

A implementação deve produzir, no mínimo, estes artefatos:

### 12.1 Driver do codec mínimo

Arquivo sugerido:

* `snd-soc-fpgafft-codec.c`

### 12.2 Overlay

Arquivo sugerido:

* `fpga-i2s-rx-32x2-slave-overlay.dts`

### 12.3 Overlay compilado

Arquivo gerado:

* `fpga-i2s-rx-32x2-slave.dtbo`

### 12.4 Documentação

Arquivo sugerido:

* `docs/rpi_fpgafft_asoc_overlay.md`

### 12.5 Script de instalação

Arquivo sugerido:

* `install_fpgafft_overlay.sh`

---

## 13. Script de instalação

## 13.1 Objetivo

O projeto deve incluir um script de instalação para reduzir erro operacional e padronizar a ativação do driver/overlay.

## 13.2 O que o script deve fazer

O script de instalação deve:

* verificar se está rodando com privilégios adequados;
* localizar o `config.txt` correto:

  * `/boot/firmware/config.txt`
  * ou `/boot/config.txt`
* criar backup versionado do `config.txt`;
* copiar o `.dtbo` para o diretório correto de overlays;
* garantir a linha `dtoverlay=fpga-i2s-rx-32x2-slave`;
* opcionalmente garantir `dtparam=i2s=on`;
* executar `depmod -a` se o módulo do codec mínimo for instalado;
* orientar o usuário a reiniciar;
* imprimir passos de validação pós-reboot.

A documentação oficial do Raspberry Pi confirma que `config.txt` é o local de configuração para `dtoverlay`. ([Raspberry Pi][1])

## 13.3 O que o script não deve fazer

O script não deve:

* assumir que a sound card já está válida só por copiar o overlay;
* tentar validar o protocolo raw/tagged;
* ocultar erros de instalação;
* forçar autodetecção do melhor device ALSA.

## 13.4 Estrutura lógica esperada do script

O script deve conter funções como:

* `find_config_file`
* `find_overlay_dir`
* `backup_config`
* `install_dtbo`
* `ensure_dtoverlay_line`
* `print_post_reboot_checklist`

## 13.5 Checklist pós-reboot que o script deve imprimir

Após a instalação, o script deve orientar:

```bash
arecord -l
aplay -l
pinctrl get 18
pinctrl get 19
pinctrl get 20
pinctrl get 21
dmesg -l err,warn
```

Além disso, pode sugerir olhar `/proc/device-tree` e testar o overlay em runtime com `dtoverlay` durante o debug; isso é uma prática útil mencionada em orientação de debug da comunidade Raspberry Pi. ([Fóruns Raspberry Pi][3])

---

## 14. Procedimento de build

## 14.1 Compilação do overlay

O overlay deve ser compilado com `dtc -@`, para preservar símbolos/phandles adequados a overlays.

Exemplo:

```bash
dtc -@ -I dts -O dtb -o fpga-i2s-rx-32x2-slave.dtbo fpga-i2s-rx-32x2-slave-overlay.dts
```

## 14.2 Instalação do overlay

O `.dtbo` deve ser copiado para:

* `/boot/overlays`
* ou `/boot/firmware/overlays`

dependendo da imagem instalada.

## 14.3 Ativação

No `config.txt`:

```ini
dtoverlay=fpga-i2s-rx-32x2-slave
```

---

## 15. Procedimento de validação

## 15.1 Validação do boot e do Device Tree

Após reinício:

* confirmar presença da sound card em `arecord -l`
* confirmar ausência de erros críticos em `dmesg -l err,warn`
* confirmar presença das mudanças esperadas em `/proc/device-tree`, se necessário. A orientação de debug de overlays do Raspberry Pi recomenda exatamente essas verificações. ([Fóruns Raspberry Pi][3])

## 15.2 Validação do pinmux

Executar:

```bash
pinctrl get 18
pinctrl get 19
pinctrl get 20
pinctrl get 21
```

## 15.3 Validação do ALSA

Executar:

```bash
arecord --dump-hw-params -D hw:X,Y
```

## 15.4 Validação do conteúdo bruto

Usar o utilitário em C que imprime:

```text
0xLEFTWORD 0xRIGHTWORD
```

Essa validação é obrigatória antes de voltar ao parser Python.

## 15.5 Validação semântica posterior

Somente após o transporte estar correto:

* testar raw capture
* testar tagged mode
* testar debug replay
* testar FFT parser

---

## 16. Critérios de aceitação

A implementação será aceita quando:

1. o codec mínimo carregar e registrar corretamente seu DAI;
2. o overlay carregar no boot sem erro relevante;
3. a sound card aparecer em `arecord -l`;
4. o ALSA aceitar captura estéreo `S32_LE`;
5. o utilitário C ler words coerentes com padrões conhecidos transmitidos pela FPGA;
6. raw e tagged funcionarem sobre o mesmo transporte;
7. a camada kernel/overlay deixar de ser suspeita principal na análise de erro.

---

## 17. Riscos técnicos

### Risco 1 — foco excessivo no overlay

Mesmo com overlay e codec corretos, o stream ainda pode estar errado por causa do RTL I2S da FPGA.

### Risco 2 — taxa nominal incomum

A taxa de 48 828,125 Hz pode exigir cuidado na abertura do ALSA e na validação.

### Risco 3 — falsa sensação de sucesso

A sound card pode aparecer e abrir, mas ainda assim o stream bruto pode estar semanticamente quebrado. Por isso o utilitário C e a validação em hex são mandatórios.

### Risco 4 — instalação parcial

Instalar só o overlay sem o codec mínimo, ou vice-versa, não fecha a topologia ASoC.

---

## 18. Ordem recomendada de implementação

### Fase 1

Escrever a documentação técnica final.

### Fase 2

Implementar o codec mínimo.

### Fase 3

Implementar o `.dts` do overlay.

### Fase 4

Implementar o script de instalação.

### Fase 5

Instalar e validar:

* boot
* ALSA
* pinmux
* utilitário C

### Fase 6

Só depois retomar o parser Python e o protocolo tagged.

---

## 19. Síntese executiva

A solução oficial do projeto passa a ser:

* **um codec mínimo próprio**, sem plano de controle;
* **um overlay próprio baseado em `simple-audio-card`**;
* **FPGA sempre como mestre de clock**;
* **Pi sempre como receptor I2S**;
* **ALSA expondo estéreo `S32_LE`**;
* **user space responsável por toda interpretação do protocolo**;
* **um script de instalação oficial** para copiar o `.dtbo`, editar `config.txt`, orientar reboot e checklist de validação.

Essa arquitetura é coerente com o mecanismo de overlays do Raspberry Pi e com o binding do `simple-audio-card`. ([Raspberry Pi][1])

---

[1]: https://www.raspberrypi.com/documentation/computers/config_txt.html?utm_source=chatgpt.com "config.txt - Raspberry Pi Documentation"
[2]: https://www.kernel.org/doc/Documentation/devicetree/bindings/sound/simple-card.txt?utm_source=chatgpt.com "simple-card.txt"
[3]: https://forums.raspberrypi.com/viewtopic.php?t=387792&utm_source=chatgpt.com "How to debug overlays?"
