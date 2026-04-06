> Nota de historico: este documento registra a fase de comparacao entre overlays.
> A solucao oficial implementada no projeto agora esta em
> `docs/overlay_implemenation_plan.md` e em
> `rpi3b_i2s_fft/asoc/fpga-i2s-rx-32x2-slave-overlay.dts`.
>
Para o que vocês estão fazendo, eu dividiria os overlays em três níveis de adequação:

## 1. Overlay ideal: **um overlay próprio do projeto**

O melhor cenário é usar um overlay seu, baseado em **`simple-audio-card`**, configurado para o dispositivo externo ser o mestre de `BCLK` e `LRCLK`, deixando o Raspberry Pi como **slave/clock consumer**. Esse é o desenho mais alinhado com “FPGA gera os clocks, Pi só captura”. A base conceitual existe no ecossistema Raspberry Pi e aparece em overlays/documentação que usam `simple-audio-card`; além disso, a documentação oficial deixa claro que `dtoverlay` no `config.txt` é a forma correta de ativar esse caminho de hardware no boot. ([GitHub][1])

### Por que esse é o ideal

Porque você controla explicitamente:

* o papel master/slave,
* o nome da sound card ALSA,
* o formato esperado,
* e elimina a dependência de “reaproveitar” um overlay feito para outro HAT.

## 2. Overlay candidato de laboratório: **`ugreen-dabboard`**

Esse overlay é interessante porque a própria descrição dele diz que é baseado em `simple-audio-card` e tem a característica de ser configurado para usar o codec como **master I2S device**. Isso o torna um candidato mais coerente com clock externo do que overlays de HATs de voz genéricos. ([GitHub][1])

### Limitação

Ele ainda não é “feito para sua FPGA”; é apenas um overlay cuja filosofia de clocking parece mais próxima do seu caso.

## 3. Fallback pragmático: **`googlevoicehat-soundcard`**

Esse é o que vocês já estão usando. Ele pode servir para bring-up e testes rápidos, mas eu não trataria como solução final. Ele é um overlay de uma placa específica, não uma declaração explícita do teu contrato FPGA→Pi, e não foi feito pensando no teu serializer/tagged protocol. ([GitHub][1])

---

# Minha recomendação objetiva

Para o teu projeto, a ordem certa é:

**melhor opção:** overlay próprio `simple-audio-card` para Pi slave
**segunda opção para experimento:** `ugreen-dabboard`
**terceira opção/fallback:** `googlevoicehat-soundcard` ([GitHub][1])

---

# Como testar overlays de forma correta

O erro comum é testar overlay só com:

* “apareceu em `arecord -l`”
* “o script abriu”

Isso é insuficiente. O teste precisa ser em camadas.

## Etapa 1 — verificar se o overlay carregou e criou a sound card

Depois de editar o `config.txt` e rebootar, verifique:

```bash
arecord -l
aplay -l
```

Se o overlay estiver funcional, uma sound card de captura deve aparecer. A documentação oficial do Raspberry Pi e a documentação de overlays convergem nesse fluxo: `dtoverlay` no boot configura o hardware e depois o ALSA passa a ver o dispositivo. ([Fóruns Raspberry Pi][2])

## Etapa 2 — verificar pinmux

Cheque se os pinos foram realmente postos na função PCM/I2S:

```bash
pinctrl get 18
pinctrl get 19
pinctrl get 20
pinctrl get 21
```

Se o sistema não tiver `pinctrl`, use `raspi-gpio get ...`. Isso não prova captura correta, mas prova que o overlay mexeu no pinmux.

## Etapa 3 — verificar se o device ALSA aceita o formato

Use:

```bash
arecord --dump-hw-params -D hw:X,Y
```

ou faça um teste de captura curta. O que você quer confirmar aqui é se o caminho está realmente aceitando algo compatível com:

* `S32_LE`
* 2 canais
* taxa nominal esperada

## Etapa 4 — verificar clocks físicos

Meça no Pi, não só na FPGA:

* `GPIO18` = `BCLK`
* `GPIO19` = `LRCLK`

Se o overlay/device estiver aberto, mas os clocks externos não estiverem chegando limpos ao Pi, nada acima disso vai ser confiável.

## Etapa 5 — verificar conteúdo bruto

Use teu utilitário em C ou a captura raw do `analyzer_from_fpga_fft.py` para ver:

* se os words batem com o padrão esperado,
* se left/right estão corretos,
* se os bits de tag aparecem onde deveriam.

---

# O que seria um teste comparativo bom entre overlays

Eu faria uma matriz pequena:

### Overlay A

`googlevoicehat-soundcard`

### Overlay B

`ugreen-dabboard`

### Overlay C

overlay próprio do projeto, quando estiver pronto

Para cada um, repetir exatamente o mesmo protocolo:

1. editar `config.txt`
2. reboot
3. `arecord -l`
4. `pinctrl get 18 19 20 21`
5. rodar utilitário C com device fixo
6. rodar captura raw
7. rodar debug tagged com a mesma FPGA e o mesmo bitstream

E comparar:

* sound card apareceu?
* captura abre sem erro?
* overrun aparece?
* words hex fazem sentido?
* `tag_mismatch` cai?
* `fft_run_lengths` ficam longos?

---

# Critérios objetivos para dizer que um overlay é melhor

O melhor overlay não é o que “abre”, e sim o que produz:

* device ALSA estável,
* captura sem overrun frequente,
* words hex compatíveis com o padrão transmitido,
* poucos ou zero `tag_mismatch`,
* `reserved_nonzero_words` muito baixos,
* `fft_run_lengths` longos e coerentes.

Se dois overlays “funcionarem”, eu escolheria o que entregar o stream mais semanticamente limpo.

---

# Sinal de alerta importante

Se um overlay:

* cria sound card,
* deixa `arecord` abrir,
* mas o conteúdo bruto continua semanticamente quebrado,

isso **não prova** que o overlay é culpado. Pode continuar sendo bug do transmissor. O overlay só entra como principal suspeito quando:

* o mesmo transmissor, medido no fio, parece correto,
* mas o Pi continua reconstruindo words errados.

---

# Melhor estratégia de curto prazo

Hoje eu faria assim:

1. manter `googlevoicehat-soundcard` como baseline atual
2. testar `ugreen-dabboard` como candidato mais promissor pronto
3. em paralelo, preparar um overlay próprio `simple-audio-card` para o caso FPGA-master / Pi-slave

Isso evita ficar preso demais a um overlay de voice HAT que não foi feito para tua topologia. ([GitHub][1])

---

# Em uma frase

Para o teu caso, o overlay ideal é **um overlay customizado baseado em `simple-audio-card` com o dispositivo externo como mestre de clock**; `ugreen-dabboard` vale como experimento por já seguir essa filosofia, e `googlevoicehat-soundcard` deve ser tratado só como fallback de bring-up. ([GitHub][1])

Posso montar um **plano de testes passo a passo** para você executar cada overlay e registrar os resultados em tabela.

[1]: https://github.com/Hexxeh/rpi-firmware/blob/master/overlays/README?utm_source=chatgpt.com "rpi-firmware/overlays/README at master"
[2]: https://forums.raspberrypi.com/viewtopic.php?t=347811&utm_source=chatgpt.com "DT overlay for I2S loads correct modules, but sound card ..."
