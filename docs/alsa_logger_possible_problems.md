Se esse programa em C **não ler os valores esperados**, os erros podem vir de quatro camadas diferentes:

1. **transmissor FPGA**
2. **sinal físico/I2S**
3. **Raspberry Pi + driver/overlay/ALSA**
4. **hipótese errada sobre o protocolo**

O jeito certo de depurar é isolar em qual camada o erro nasce.

# 1. Erros no transmissor FPGA

Essa é hoje uma das fontes mais fortes de suspeita no teu caso.

## a) Left/right trocados ou desalinhados

O host espera frames estéreo assim:

* left word
* right word

Se o RTL estiver trocando o payload ativo na borda errada, o Pi pode receber algo como:

* right de um item
* left do próximo

Aí o programa em C vai imprimir words válidos em hexadecimal, mas **não os words esperados nos pares esperados**.

### Sintoma típico

* os hex parecem “quase certos”
* mas left e right não combinam
* ou o padrão esperado aparece deslocado de uma linha para outra

---

## b) Tag bits ou payload empacotados errado

Se a FPGA está montando:

* bits `[31:30]` = tag
* bits `[17:0]` = payload

mas na prática:

* a tag foi parar em outra posição,
* o payload foi sign-extended errado,
* os bits reservados não estão zerados,

o programa em C vai mostrar words diferentes do esperado.

### Sintoma típico

Você esperava algo como:

* `0x80000001`
* `0x80000002`

mas aparece:

* `0x40000001`
* `0x20000001`
* ou palavras com miolo não zero

---

## c) Ordem dos bits no serializer

O serializer pode estar enviando:

* MSB first corretamente,
  ou
* invertido,
  ou
* com deslocamento de 1 bit

Se houver deslocamento, os hex no Pi ficam todos “parecidos mas errados”.

### Sintoma típico

Um valor que deveria ser:

* `0x80000001`

vira algo como:

* `0x40000000`
* `0x00000003`
* ou qualquer valor sistematicamente deslocado

---

## d) Atualização de payload no meio da palavra

Se `SD` ou o registrador de saída mudam combinacionalmente durante a transmissão da palavra, o host pode reconstruir words corrompidos.

### Sintoma típico

* palavras instáveis
* valores mudando sem padrão
* tags inconsistentes
* lixo aparente mesmo com clocks certos

---

# 2. Erros de sinal físico / temporização I2S

Mesmo que o RTL “pareça certo” na simulação, o problema pode estar no timing físico.

## a) Timing Philips I2S errado

No I2S Philips:

* `WS/LRCLK` muda
* e o `MSB` da próxima palavra vem **um bit clock depois**

Se a FPGA colocar o MSB no mesmo instante da troca de `WS`, o Pi pode ler tudo deslocado.

### Sintoma típico

Todos os words ficam errados de forma sistemática, como se faltasse ou sobrasse 1 bit.

---

## b) Número errado de bits por slot

O Pi/driver pode estar esperando 32 bits por slot, mas a FPGA pode estar efetivamente entregando:

* 24 bits justificados de outro jeito
* 18 bits sem sign extension apropriada
* ou slot width diferente

### Sintoma típico

Os valores aparecem truncados, repetidos ou com padding inesperado.

---

## c) Clock ruim ou instável

Se:

* `BCLK` estiver com ruído,
* `LRCLK` estiver errado,
* duty cycle estiver ruim,
* bordas estiverem degradadas,

o Pi pode amostrar errado.

### Sintoma típico

* às vezes lê certo
* às vezes lixo
* comportamento intermitente
* overrun ou instabilidade junto

---

## d) Níveis elétricos

Se houver:

* problema de 3,3 V
* ground ruim
* cabo ruim
* acoplamento
* mau contato

o stream pode chegar corrompido.

### Sintoma típico

Comportamento não determinístico, dependente de montagem/fiação.

---

# 3. Erros no Raspberry Pi / ALSA / overlay

Mesmo com a FPGA correta, o Pi ainda pode estar interpretando errado.

## a) Overlay inadequado

Se o overlay não estiver realmente apropriado para:

* Pi em slave
* clocks externos vindos da FPGA

o ALSA pode até abrir o device, mas a captura pode ficar semanticamente errada ou instável.

### Sintoma típico

* device aparece
* programa roda
* mas os valores não batem ou os frames ficam estranhos

---

## b) Device ALSA errado

Se o programa em C estiver abrindo o card errado, ele vai imprimir outra fonte de dados.

### Sintoma típico

* valores não têm nada a ver com o protocolo
* podem parecer ruído ou zeros
* ou vir de outro dispositivo

Por isso, para debug, sempre fixe `hw:X,Y`.

---

## c) Formato ALSA não bate com o real

O código C abre:

* `S32_LE`
* 2 canais
* interleaved

Se o device real estiver configurado de outro jeito por baixo, pode haver mismatch.

### Sintoma típico

* cada word parece deslocado
* canais trocados
* bytes interpretados errado

---

## d) Overrun

Se o sistema estiver sofrendo overrun, o programa pode perder trechos do stream.

### Sintoma típico

* algumas linhas boas
* depois lixo
* depois volta
* ou stream interrompido

No utilitário em C isso tende a ser bem menos provável do que no logger Python, mas ainda pode acontecer.

---

# 4. Erros na sua expectativa, não necessariamente no hardware

Essa camada é muito importante.

## a) Você está esperando o word errado

Talvez o stream real não esteja enviando exatamente:

* left = real
* right = imag

ou:

* o transmissor esteja mandando BFPEXP / FFT / IDLE em outra ordem

### Sintoma típico

Os valores são consistentes, mas não com a hipótese mental inicial.

---

## b) Tag shift errado

Você pode estar esperando:

* tag em `[31:30]`

mas na prática ela pode estar em:

* `[30:29]`
* ou outro offset

### Sintoma típico

Os hex parecem não fazer sentido até reinterpretar os bits de outra forma.

---

## c) Right/left semanticamente trocados

Talvez:

* o Pi esteja lendo left/right corretamente,
* mas a FPGA decidiu usar:

  * left = imag
  * right = real

### Sintoma típico

Os valores fazem sentido, mas invertidos.

---

## d) Endianness mental errada

O ALSA entrega `S32_LE`, mas quando você olha hexadecimal, é fácil confundir:

* ordem dos bytes na memória
  com
* valor inteiro reconstruído

No `printf("0x%08X")`, o que você vê é o **inteiro já reconstruído**, não o byte stream cru.

---

# Como separar as causas na prática

## Caso 1: no analisador lógico já está errado

Então o problema é:

* FPGA
* serializer
* timing I2S
* elétrica

e não ALSA.

---

## Caso 2: no analisador lógico está certo, mas o C mostra errado

Então o problema está mais provavelmente em:

* overlay
* driver
* formato ALSA
* hipótese de enquadramento do host

---

## Caso 3: o C mostra um padrão consistente, mas diferente do esperado

Então o problema pode ser:

* tag shift errado
* left/right trocados
* protocolo semântico diferente do assumido

---

## Caso 4: o C mostra valores instáveis/intermitentes

Então suspeite de:

* clock/timing físico
* elétrica
* overrun
* RTL mudando payload no instante errado

---

# Melhor estratégia de debug

Eu faria nesta ordem:

## 1. Mandar um padrão fixo muito simples pela FPGA

Por exemplo, durante alguns segundos:

* left = `0x40000001`
* right = `0x40000002`

ou qualquer padrão fácil de reconhecer.

Se o programa em C não imprimir isso exatamente, o problema está antes do pipeline alto nível.

---

## 2. Testar com padrões alternados

Exemplo:

* quadro 0: `left=0x40000011`, `right=0x40000022`
* quadro 1: `left=0x80000033`, `right=0x80000044`

Isso ajuda a detectar:

* troca de canais
* deslocamento entre frames
* tags em posição errada

---

## 3. Verificar com analisador lógico

Conferir:

* BCLK
* LRCLK
* DIN
* instante de troca de `WS`
* MSB um bit depois da troca

---

## 4. Rodar o utilitário C com device fixo

Nada de autodetecção.

---

## 5. Só depois voltar ao parser Python

Porque o C elimina uma boa parte das ambiguidades do pipeline.

---

# Heurística rápida de diagnóstico

Se o programa C mostrar:

### Sempre zero

Pode ser:

* sem sinal real
* device errado
* overlay errado
* FPGA não transmitindo

### Valores válidos mas left/right errados

Pode ser:

* bug na fronteira de slot
* canais trocados
* atualização no momento errado

### Valores sempre deslocados

Pode ser:

* timing Philips errado
* 1 bit de deslocamento
* tag shift/payload shift errados

### Valores aleatórios/intermitentes

Pode ser:

* elétrica
* clock ruim
* overrun
* serializer instável

### Valores consistentes, mas não com o esperado

Pode ser:

* hipótese errada sobre protocolo
* tag/slot/order diferentes do assumido

---

# Resumo

Se o programa em C não ler os valores esperados, os erros podem vir de:

* **FPGA/RTL**: empacotamento, serializer, borda errada, left/right, tag, payload
* **I2S físico**: clocks, timing Philips, slot width, elétrica
* **Pi/ALSA**: overlay, formato, device errado, overrun
* **hipótese errada**: tag shift, ordem semântica, interpretação do protocolo

O teste mais poderoso é usar a FPGA para transmitir um padrão fixo extremamente reconhecível e verificar se ele aparece exatamente igual no utilitário em C. Se não aparecer, aí dá para localizar em qual camada a distorção entrou.
