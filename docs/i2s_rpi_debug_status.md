# Status do Debug I2S no Raspberry Pi

Este documento registra o estado atual da investigacao do caminho
FPGA `I2S TX` -> Raspberry Pi `ASoC/ALSA`.

O objetivo aqui e preservar o historico tecnico que levou ao estado parcial
atual: o barramento no fio ja ficou consistente, o host ja consegue recuperar
os words corretos em combinacoes especificas, mas ainda existe instabilidade
residual na abertura do stream ALSA.

## Contrato de diagnostico usado na investigacao

O topo de diagnostico foi mantido emitindo um padrao fixo e facilmente
reconhecivel no host:

- `left = 0x80015555`
- `right = 0x8000AAAB`
- `bfpexp = 0x40000012`

Esse contrato foi essencial porque permitiu distinguir:

- erro de endianess,
- erro de framing/slot no I2S,
- troca de canais,
- perda do bit mais significativo,
- e lock em fase errada no lado do Raspberry Pi.

## Sequencia da investigacao

### 1. Sintoma inicial no host

No inicio da investigacao, o Raspberry Pi nao recebia os words esperados.
Em vez de observar diretamente `0x80015555 / 0x8000AAAB`, apareciam valores
como:

- `0x00155570 / 0x002AAAB0`
- `0xAAB80015 / 0x5558000A`
- `0x02AAAB00 / 0x01555700`

Esses padroes indicavam deslocamento/rotacao de bits dentro do slot, e nao um
simples problema de endianess ou parser.

### 2. Erro confirmado no transmissor I2S da FPGA

O primeiro erro real encontrado estava no serializer
[`rtl/frontend/i2s_fft_tx_adapter.sv`](../../../rtl/frontend/i2s_fft_tx_adapter.sv).

O problema era temporal:

- `WS` e `SD` nao estavam sendo entregues ao pino com setup suficientemente
  robusto para o `bcm2835-i2s` em slave mode;
- a mudanca de `WS` estava muito proxima da borda relevante do `BCLK`;
- isso quebrava a interpretacao Philips I2S no host, mesmo quando o stream
  parecia "quase certo" em simulacao.

### 3. Correcao aplicada no RTL

O `i2s_fft_tx_adapter` foi corrigido para:

- exigir `CLOCK_DIV >= 2`;
- antecipar `WS` antes da borda negativa relevante do `BCLK`;
- manter `SD` sendo preparado no semiperiodo correto;
- preservar o contrato de tags e payload do stream tagged.

O resultado dessa etapa foi importante: o problema deixou de parecer puramente
um bug "misterioso" de software e passou a responder melhor ao formato/polaridade
do lado do Pi.

### 4. Evidencia do barramento no fio

Depois da correcao do TX, a captura no Analog Discovery passou a mostrar o
barramento com decode correto dos words de diagnostico.

Isso foi o pivô da investigacao:

- o fio passou a parecer bom;
- o problema restante passou a ser "o que o Raspberry Pi faz com esse fio",
  e nao mais "o que a FPGA esta serializando".

Em outras palavras:

- **sinal no fio**: havia evidencia de palavras corretas;
- **interpretacao no host**: ainda podia abrir em fase errada.

### 5. Pivô para ASoC/overlay/driver no Raspberry Pi

Com o fio aparentemente correto, a investigacao migrou para:

- overlay `simple-audio-card`,
- codec stub `snd-soc-fpgafft-codec`,
- formato `DAIFMT`,
- e comportamento do `bcm2835-i2s` em slave mode.

Foram adicionados:

- logs de `set_fmt` no codec;
- variantes de overlay para combinacoes de formato/inversao;
- e, mais tarde, um realinhador no software Python para recuperar o stream
  correto quando o ALSA abria em fase errada.

## Erros encontrados e classificados

### Erros confirmados

1. **Erro temporal original no serializer TX**

O transmissor I2S original nao dava uma relacao temporal suficientemente limpa
entre `WS`, `BCLK` e `SD` para o receptor do Raspberry Pi.

2. **Interpretacoes incorretas do `bcm2835-i2s` para o mesmo fio**

Ao variar `format` e `inversion`, o mesmo barramento gerava padroes
dominantes diferentes no host. Isso mostrou que o problema residual estava na
forma como o Pi enquadrava os 32 bits, e nao no payload em si.

3. **Lock instavel do ALSA em fases diferentes**

Mesmo mantendo overlay/bitstream constantes, repetidas aberturas de `arecord`
podiam "travar" em fases diferentes do mesmo stream. Esse comportamento foi
reproduzido varias vezes durante a investigacao.

4. **`overrun` do logger Python**

O `fft_i2s_logger.py` mostrou `overrun` em varias execucoes. Isso nao explica a
origem do erro de framing, mas adiciona ruido operacional e dificulta a
reproducao do estado do link.

5. **Casos com words corretos mas canais trocados**

Algumas combinacoes chegaram a reconstruir os words certos, porem com ordem
`left/right` invertida na abertura observada.

### Hipotese atual aberta

A hipotese tecnica mais forte para o problema restante e:

- o `bcm2835-i2s` em slave mode ainda abre o stream em fases diferentes na
  inicializacao da captura;
- o barramento pode estar aceitavel no fio, mas o host ainda nao abre de modo
  deterministicamente alinhado em todas as execucoes.

## Matriz resumida de experimentos no Pi

Tabela compacta dos experimentos mais importantes ja observados.
Os valores abaixo representam o padrao dominante de uma execucao relevante de
cada variante, nao uma garantia de estabilidade entre reaberturas.

| Overlay / formato | `set_fmt raw` | Padrao dominante capturado | Leitura do resultado |
| --- | --- | --- | --- |
| `I2S + NB_NF` | `0x1001` | ex.: `0xAAB00015 / 0x5570002A` | Framing ainda incorreto; a fase variou entre aberturas. |
| `I2S + IB_NF` | `0x1301` | `0x55556000 / 0x2AAAE000` | A inversao de `BCLK` mudou claramente o erro, mas nao corrigiu o stream. |
| `I2S + NB_IF` | `0x1201` | `0x70002AAA / 0xB0001555` | `frame-inversion` mudou o enquadramento, mas continuou errado. |
| `I2S + IB_IF` | `0x1401` | `0xAC000555 / 0x5C000AAA` | Continua incorreto; o problema nao era resolvido so com polaridade. |
| `LEFT_J + NB_NF` | `0x1003` | `0x0002AAAB / 0x00015557` | `data_delay = 0` aproximou o payload util, mas ainda com deslocamento. |
| `LEFT_J + IB_NF` | `0x1303` | melhor abertura: `0x80015555 / 0x8000AAAB` | Variante mais promissora; chegou ao par correto em captura crua, mas ainda pode abrir em outra fase. |
| `LEFT_J + NB_IF` | `0x1203` | melhor abertura: `0x8000AAAB / 0x80015555` | Words corretos com canais trocados; reaberturas ainda nao ficaram deterministicas. |

## Mudancas feitas para chegar ao estado atual

### RTL

- Correcao temporal do serializer em
  [`rtl/frontend/i2s_fft_tx_adapter.sv`](../../../rtl/frontend/i2s_fft_tx_adapter.sv).
- Ajuste do `WS` para chegar ao pino com antecedencia suficiente ao receptor.
- Preservacao do contrato tagged `{tag, zeros reservados, payload}`.

### Overlay e codec no Pi

- Instrumentacao do codec
  [`rpi3b_i2s_fft/asoc/snd-soc-fpgafft-codec.c`](../rpi3b_i2s_fft/asoc/snd-soc-fpgafft-codec.c)
  para registrar `set_fmt`, `set_tdm_slot` e `hw_params`.
- Criacao de uma matriz de overlays experimentais em
  [`rpi3b_i2s_fft/asoc/`](../rpi3b_i2s_fft/asoc/):
  - `I2S + NB_NF`
  - `I2S + IB_NF`
  - `I2S + NB_IF`
  - `I2S + IB_IF`
  - `LEFT_J + NB_NF`
  - `LEFT_J + IB_NF`
  - `LEFT_J + NB_IF`
  - `LEFT_J + IB_IF`

### Scripts Python

- O `fft_i2s_logger.py` foi ajustado para gravar words em hexadecimal.
- O receiver Python recebeu um realinhador de stream em
  [`rpi3b_i2s_fft/i2s_stream.py`](../rpi3b_i2s_fft/i2s_stream.py).
- O realinhador usa:
  - campo de tag,
  - zeros reservados,
  - igualdade dos words de `BFPEXP`,
  - e possibilidade de `swap` de canais
  para encontrar a fase mais plausivel do stream recebido.
- O `fpga_fft_adapter.py` e o `fft_i2s_logger.py` passaram a reutilizar esse
  realinhador antes de interpretar/gravar os dados.

## Estado atual recomendado para continuidade

O estado mais promissor ate aqui e:

- overlay: `fpga-leftj-rx-32x2-slave-bitclock-inv`
- formato efetivo: `LEFT_J + IB_NF`
- comportamento observado no `arecord` cru:
  - em aberturas limpas, o par dominante ja chegou como
    `0x80015555 / 0x8000AAAB`
- comportamento observado no `fft_i2s_logger`:
  - com o realinhador Python, o logger passou a recuperar dominantemente os
    words corretos em varias execucoes
- limitacao restante:
  - a abertura do device ALSA ainda nao ficou deterministicamente estavel;
  - ainda podem ocorrer fases diferentes e, em algumas execucoes, ambiguidade
    de ordem `left/right`.

Em resumo:

- **bom o suficiente neste estagio**:
  - o barramento no fio nao e mais o principal suspeito;
  - o host ja consegue recuperar o conteudo correto em cenarios reais;
  - existe caminho tecnico claro para continuar.
- **ainda nao resolvido**:
  - estabilidade final do lock de abertura no ALSA/ASoC;
  - ordem de canal totalmente deterministica em todas as aberturas;
  - `overrun` do logger em validacoes longas.

## Proximo problema a resolver

O proximo alvo deve ser a ultima camada de estabilidade do lado do Pi:

1. estabilizar a abertura do stream ALSA;
2. eliminar a ambiguidade de ordem de canal;
3. reduzir `overrun` do logger durante validacao longa.

Isso deve ser tratado como problema de inicializacao/sincronismo do host,
nao como regressao do payload do barramento.
