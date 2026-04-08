# Raspberry Pi FFT Runbook Alinhado ao Contrato do TB

Este documento consolida os comandos prontos para uso do lado Python e a
referencia local do testbench Verilog. O objetivo e executar o Raspberry Pi
seguindo exatamente o contrato validado no `top_level_test`.

## Contrato validado no TB

O stream tagged validado localmente segue esta sequencia:

- `idle*`
- `bfpexp_hold_pairs x BFPEXP` com `tag=1` e `packet_index=0 .. bfpexp_hold_pairs-1`
- `512 x FFT` com `tag=2` e `packet_index=512 .. 1023`

O receptor Python deve ignorar o `idle` inicial e so aceitar um novo frame
de FFT depois de observar o preambulo completo de `BFPEXP` em modo estrito.
Quando um par `FFT` se perde no link, o receptor agora usa `packet_index` para
preencher com zero apenas o bin faltante, sem desalinhamento do restante da
janela.

Configuracao de tags validada:

- `packet_index_shift = 22`
- `packet_index_bits = 10`
- `fft_packet_index_base = 512`
- `tag_shift = 20`
- `tag_mask = 0x3`
- `payload_bits = 18`
- `tag_idle = 0`
- `tag_bfpexp = 1`
- `tag_fft = 2`
- bits reservados `[19:18] = 0`

## Referencia local do exemplo fixo

Para o `example_0 = sine_1k`, com `Fs = 48828 Hz` e `N = 512`:

- o pico principal esperado fica perto do bin `10`
- a frequencia correspondente fica em aproximadamente `953.67 Hz`
- vazamento esperado aparece ao redor de `9/11` e `10/12`, conforme a janela
  e a quantizacao do sistema

Na validacao local do TB, o burst util saiu correto:

- `128` pares `BFPEXP` indexados de `0` a `127`
- `512` pares `FFT` indexados de `512` a `1023`
- `0` mismatches nos frames uteis do transmissor
- bins verificados tambem apos a desserializacao I2S do `top_level_test`

## Onde rodar no Raspberry Pi

No Raspberry Pi, use a pasta:

```bash
cd ~/Desktop/other/ACES-RPi-interface/rpi3b_i2s_fft
```

Os comandos abaixo assumem que o ambiente da pasta ja foi configurado e que o
projeto usa a virtualenv `.venv`.

## Comandos prontos para uso no Raspberry Pi

### 1. Analisador principal em modo estrito

Este e o comando recomendado para operacao normal. Ele segue exatamente o
contrato do TB e so inicia um frame de FFT depois de ver `128` pares BFPEXP.

```bash
cd ~/Desktop/other/ACES-RPi-interface/rpi3b_i2s_fft
.venv/bin/python analyzer_from_fpga_fft.py \
  --strict-sync \
  --rate 48828 \
  --frame-bins 512 \
  --useful-bins 256 \
  --packet-index-shift 22 \
  --packet-index-bits 10 \
  --fft-packet-index-base 512 \
  --tag-shift 20 \
  --tag-mask 0x3 \
  --payload-bits 18 \
  --tag-idle 0 \
  --tag-bfpexp 1 \
  --tag-fft 2
```

O que esperar:

- `idle` inicial pode durar bastante tempo e isso e normal
- o buffer so passa a encher quando aparece o primeiro burst valido
- `Enter` salva `evento.npy` e `fft.npy`

Defaults do preset `--strict-sync`:

- modo tagged habilitado
- `bfpexp-hold-pairs = 128`
- `loss-tolerance-pairs = 3`
- `allow-fft-without-bfpexp = false`

### 2. Analisador em modo tolerante para attach no meio do burst

Use este modo apenas quando a captura for iniciada no meio do stream e voce
precisar permitir sincronizacao sem o preambulo completo.

```bash
cd ~/Desktop/other/ACES-RPi-interface/rpi3b_i2s_fft
.venv/bin/python analyzer_from_fpga_fft.py \
  --tolerant-sync \
  --rate 48828 \
  --frame-bins 512 \
  --useful-bins 256 \
  --packet-index-shift 22 \
  --packet-index-bits 10 \
  --fft-packet-index-base 512 \
  --tag-shift 20 \
  --tag-mask 0x3 \
  --payload-bits 18 \
  --tag-idle 0 \
  --tag-bfpexp 1 \
  --tag-fft 2
```

Defaults do preset `--tolerant-sync`:

- modo tagged habilitado
- `bfpexp-hold-pairs = 128`
- `loss-tolerance-pairs = 3`
- `allow-fft-without-bfpexp = true`

### 3. Plot com matplotlib

Este comando monitora `fft.npy` e atualiza `fft_latest.png`. O grafico mostra:

- um espectro representativo da FFT
- um espectrograma ao longo do tempo

```bash
cd ~/Desktop/other/ACES-RPi-interface/rpi3b_i2s_fft
.venv/bin/python plotFFT.py \
  --spectrogram \
  --fft-file fft.npy \
  --rate 48828 \
  --frame-bins 512 \
  --max-freq 5000 \
  --step-hz 500 \
  --backend Agg \
  --output-file fft_latest.png
```

O que esperar para `sine_1k`:

- o frame representativo deve destacar um pico principal perto do bin `10`
- a frequencia do pico deve ficar perto de `953.67 Hz`
- o arquivo `fft_latest.png` deve ser regravado sempre que `fft.npy` mudar

### 3.1. Capturar a proxima janela FFT e plotar centrado em 0 Hz

Use este modo quando voce quiser congelar apenas uma janela de FFT e gerar o
grafico tradicional com frequencias negativas e positivas em torno de `0 Hz`.

```bash
cd ~/Desktop/other/ACES-RPi-interface/rpi3b_i2s_fft
.venv/bin/python plotFFT.py \
  --capture-window \
  --fft-file fft.npy \
  --rate 48828 \
  --frame-bins 512 \
  --max-freq 5000 \
  --backend Agg \
  --output-file fft_single_window.png
```

O que este comando faz:

- espera a proxima atualizacao de `fft.npy`
- seleciona uma unica janela FFT
- centraliza o eixo de frequencia em `0 Hz`
- grava a figura em `fft_single_window.png`

Defaults do preset `--capture-window`:

- modo `window`
- `sample-mode = latest`
- `capture-next-window = true`
- `center-zero = true`
- `output-file = fft_single_window.png`

### 3.2. Capturar uma janela unica e sobrepor a referencia mock

Quando voce tiver acesso ao CSV esperado do testbench, o mesmo plot pode
sobrepor a curva esperada:

```bash
cd /mnt/c/Users/jvcte/quadri_poli/ACES
python3 submodules/ACES-RPi-interface/rpi3b_i2s_fft/plotFFT.py \
  --capture-window \
  --fft-file submodules/ACES-RPi-interface/rpi3b_i2s_fft/fft.npy \
  --rate 48828 \
  --frame-bins 512 \
  --max-freq 5000 \
  --backend Agg \
  --output-file /tmp/fft_single_window_mock.png \
  --mock-fft-csv tb/data/top_level_test_expected_fft.csv \
  --mock-example 0
```

Nesse modo, a curva capturada sai como `capturado` e a referencia do TB sai
como `mock esperado`.

Defaults do preset `--spectrogram`:

- modo `history`
- `sample-mode = max-energy`
- `capture-next-window = false`
- `center-zero = false`
- `output-file = fft_latest.png`

### 4. Logger CSV do stream tagged

Este comando grava o stream bruto decodificado e tambem anota a fase do
contrato para cada linha do CSV.

```bash
cd ~/Desktop/other/ACES-RPi-interface/rpi3b_i2s_fft
.venv/bin/python fft_i2s_logger.py \
  --rate 48828 \
  --frame-bins 512 \
  --bfpexp-hold-pairs 128 \
  --loss-tolerance-pairs 3 \
  --packet-index-shift 22 \
  --packet-index-bits 10 \
  --fft-packet-index-base 512 \
  --tag-shift 20 \
  --tag-mask 0x3 \
  --payload-bits 18 \
  --tag-idle 0 \
  --tag-bfpexp 1 \
  --tag-fft 2 \
  --csv fft_capture_tagged.csv
```

O CSV inclui, entre outros campos:

- words crus de esquerda e direita
- `kind`
- `tag`
- `payload`
- bits reservados
- `contract_phase`
- `contract_frame`
- `contract_index`
- `packet_index`

Observacao de robustez:

- perdas pontuais que decodam como `tag_mismatch`, `idle` ou `unknown_tag`
  dentro do preambulo BFPEXP ou da janela FFT sao toleradas por default ate
  `3` pares por burst
- no receptor, essas perdas pontuais dentro da FFT sao contabilizadas como
  bins ausentes e preenchidas com zero, evitando perder a janela inteira

### 4.1. Bundle de debug offline da causa raiz

Quando for necessario depurar offline a origem de erros observados em hardware,
o fluxo recomendado agora e gerar um bundle com captura bruta e telemetria do
helper ALSA:

```bash
cd ~/Desktop/other/ACES-RPi-interface/rpi3b_i2s_fft
.venv/bin/python analyzer_from_fpga_fft.py \
  --strict-sync \
  --rate 48828 \
  --frame-bins 512 \
  --useful-bins 256 \
  --packet-index-shift 22 \
  --packet-index-bits 10 \
  --fft-packet-index-base 512 \
  --tag-shift 20 \
  --tag-mask 0x3 \
  --payload-bits 18 \
  --tag-idle 0 --tag-bfpexp 1 --tag-fft 2 \
  --debug-raw-capture channel_capture.raw \
  --debug-raw-index channel_capture.index.jsonl \
  --debug-capture-seconds 10
```

O `channel_capture.index.jsonl` agora guarda:

- metadados da sessao e do contrato decodificado
- um indice por chunk com `pair_count`, `pair_offset`, `byte_offset` e `flag_active`
- eventos do helper ALSA: `capture_session_start`, `capture_chunk`, `capture_recovery`, `capture_final`
- um resumo `capture_diagnostics` com `xruns`, `recoveries`, `partial_reads`, `queue_high_water_slots` e `queue_full_waits`

Com isso, da para diferenciar offline se a falha veio:

- do transporte I2S / alinhamento de bits / pacote corrompido
- ou do lado host, por `xrun`, recuperacao ALSA, leitura parcial ou saturacao da fila interna

## Arquivos gerados no Raspberry Pi

Arquivos principais da execucao:

- `evento.npy`
- `fft.npy`
- `fft_latest.png`
- `fft_capture_tagged.csv`
- `channel_capture.raw`
- `channel_capture.index.jsonl`
- `scenario_summary.tsv`

## Comandos locais para referencia do TB

Quando for necessario comparar o Raspberry Pi com a referencia local,
gere primeiro o caso do seno no TB:

```bash
cd /mnt/c/Users/jvcte/quadri_poli/ACES
sim/manifest/scripts/run_questa.sh top_level_test real plusargs=+TOP_LEVEL_TEST_EXAMPLE=0
```

Depois analise os dumps:

```bash
cd /mnt/c/Users/jvcte/quadri_poli/ACES
python3 utils/analyze_top_level_test_dump.py \
  --dump-dir sim/local/questa/top_level_test_real \
  --example 0
```

O que deve aparecer na referencia local:

- frontend correto do ROM ate a entrada da FFT
- `512` bins validos na saida da FFT
- tags uteis do transmissor sem mismatch
- pico dominante perto do bin `10` para o `sine_1k`

## Checklist rapido de uso

1. Suba o `analyzer_from_fpga_fft.py` com `--strict-sync`.
2. Suba o `plotFFT.py` para acompanhar `fft.npy`.
3. Se precisar inspecionar o protocolo, suba tambem o `fft_i2s_logger.py`.
4. Se precisar depurar offline a causa raiz, gere tambem `channel_capture.raw` + `channel_capture.index.jsonl`.
5. Pressione `Enter` no analisador para salvar o evento de referencia.
6. Confira se o `fft_latest.png` mostra o pico esperado perto de `953.67 Hz`.

## Observacoes importantes

- `idle` longo antes do burst util e esperado e nao indica erro.
- O problema observado anteriormente no Pi era compativel com um receptor que
  tentava montar a janela antes do primeiro preambulo BFPEXP completo.
- O fluxo Python atual foi ajustado para seguir o mesmo criterio do TB
  Verilog, que foi a referencia usada para validar o sistema.
- O bundle offline atual tambem preserva a telemetria do `alsa_logger`, entao
  um replay local passa a mostrar nao apenas os sintomas do stream, mas tambem
  se houve `xrun`, recuperacao ou pressao na fila de captura durante a aquisicao.
