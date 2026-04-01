Claro. Vou explicar o código por blocos e linha a linha, com foco no que cada parte faz no contexto do ALSA e da captura I2S.

---

## Includes

```c
#include <alsa/asoundlib.h>
```

Inclui a biblioteca ALSA para acessar dispositivos de áudio diretamente em C.

```c
#include <errno.h>
```

Inclui definições de códigos de erro do sistema, como `EPIPE`.

```c
#include <inttypes.h>
```

Inclui macros e tipos inteiros portáveis, como `PRIX32`, usado no `printf`.

```c
#include <signal.h>
```

Permite capturar sinais como `Ctrl+C` (`SIGINT`) e encerramento (`SIGTERM`).

```c
#include <stdint.h>
```

Define tipos inteiros de tamanho fixo, como `int32_t` e `uint32_t`.

```c
#include <stdio.h>
```

Funções de entrada e saída padrão, como `printf` e `fprintf`.

```c
#include <stdlib.h>
```

Funções gerais como `calloc`, `free` e `strtoul`.

```c
#include <string.h>
```

Usado aqui para `memset`.

---

## Flag global de parada

```c
static volatile sig_atomic_t g_stop = 0;
```

* `static`: visível só neste arquivo.
* `volatile`: diz ao compilador que essa variável pode mudar “por fora” do fluxo normal, como por um handler de sinal.
* `sig_atomic_t`: tipo seguro para ser alterado dentro de handler de sinal.
* `g_stop = 0`: começa com “não parar”.

Essa variável é lida no loop principal. Quando o usuário apertar `Ctrl+C`, ela vira `1`.

---

## Handler de sinal

```c
static void handle_signal(int sig) {
    (void)sig;
    g_stop = 1;
}
```

Essa função é chamada quando chega um sinal como `SIGINT`.

* `int sig`: recebe o número do sinal.
* `(void)sig;`: evita warning de variável não usada.
* `g_stop = 1;`: pede para o loop principal terminar de forma limpa.

---

## Registro dos sinais

```c
static void setup_signals(void) {
```

Define uma função auxiliar para configurar o tratamento de sinais.

```c
    struct sigaction sa;
```

Cria uma estrutura `sigaction`, usada para registrar o handler.

```c
    memset(&sa, 0, sizeof(sa));
```

Zera toda a estrutura para começar limpa.

```c
    sa.sa_handler = handle_signal;
```

Diz que, quando o sinal chegar, a função chamada será `handle_signal`.

```c
    sigemptyset(&sa.sa_mask);
```

Inicializa a máscara de sinais bloqueados durante o handler como vazia.

```c
    sigaction(SIGINT, &sa, NULL);
```

Associa `Ctrl+C` ao handler.

```c
    sigaction(SIGTERM, &sa, NULL);
```

Associa também pedido de término ao handler.

```c
}
```

Fim da função.

---

## Função para configurar hardware ALSA

```c
static int set_hw_params(
    snd_pcm_t *pcm,
    unsigned int rate,
    snd_pcm_uframes_t period_frames,
    snd_pcm_uframes_t buffer_frames
) {
```

Essa função configura o dispositivo ALSA de captura.

Parâmetros:

* `pcm`: handle do dispositivo ALSA já aberto.
* `rate`: taxa de amostragem.
* `period_frames`: tamanho do período ALSA.
* `buffer_frames`: tamanho total do buffer ALSA.

```c
    int err;
```

Variável para guardar erros retornados pelas funções ALSA.

```c
    snd_pcm_hw_params_t *hw = NULL;
```

Ponteiro para estrutura de parâmetros de hardware.

```c
    snd_pcm_hw_params_alloca(&hw);
```

Aloca essa estrutura na stack com helper do ALSA.

---

### Inicializar espaço de parâmetros

```c
    if ((err = snd_pcm_hw_params_any(pcm, hw)) < 0) {
```

Pede ao ALSA um conjunto inicial de parâmetros válidos para esse dispositivo.

```c
        fprintf(stderr, "snd_pcm_hw_params_any: %s\n", snd_strerror(err));
```

Se falhar, imprime mensagem legível do erro.

```c
        return err;
    }
```

Sai da função retornando o código de erro.

---

### Tipo de acesso: interleaved

```c
    if ((err = snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
```

Configura o modo de acesso como interleaved.

Isso significa que os samples vêm como:

* left0, right0, left1, right1, ...

e não como buffers separados por canal.

```c
        fprintf(stderr, "snd_pcm_hw_params_set_access: %s\n", snd_strerror(err));
        return err;
    }
```

Erro e retorno, se falhar.

---

### Formato: 32-bit signed little-endian

```c
    if ((err = snd_pcm_hw_params_set_format(pcm, hw, SND_PCM_FORMAT_S32_LE)) < 0) {
```

Configura o formato como `S32_LE`:

* signed
* 32 bits
* little-endian

Isso bate com o seu protocolo no host.

```c
        fprintf(stderr, "snd_pcm_hw_params_set_format: %s\n", snd_strerror(err));
        return err;
    }
```

---

### Número de canais

```c
    if ((err = snd_pcm_hw_params_set_channels(pcm, hw, 2)) < 0) {
```

Configura 2 canais: estéreo.

```c
        fprintf(stderr, "snd_pcm_hw_params_set_channels: %s\n", snd_strerror(err));
        return err;
    }
```

---

### Taxa de amostragem

```c
    if ((err = snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, 0)) < 0) {
```

Pede a taxa de amostragem mais próxima possível de `rate`.

O ALSA pode ajustar para uma taxa próxima, dependendo do dispositivo.

```c
        fprintf(stderr, "snd_pcm_hw_params_set_rate_near: %s\n", snd_strerror(err));
        return err;
    }
```

---

### Tamanho do período

```c
    if ((err = snd_pcm_hw_params_set_period_size_near(pcm, hw, &period_frames, 0)) < 0) {
```

Configura o tamanho de período do ALSA.

O período é uma unidade interna de transferência/sincronização do buffer.

```c
        fprintf(stderr, "snd_pcm_hw_params_set_period_size_near: %s\n", snd_strerror(err));
        return err;
    }
```

---

### Tamanho do buffer

```c
    if ((err = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &buffer_frames)) < 0) {
```

Configura o tamanho total do buffer interno do ALSA.

```c
        fprintf(stderr, "snd_pcm_hw_params_set_buffer_size_near: %s\n", snd_strerror(err));
        return err;
    }
```

---

### Aplicar os parâmetros

```c
    if ((err = snd_pcm_hw_params(pcm, hw)) < 0) {
```

Aplica todas as configurações no dispositivo.

```c
        fprintf(stderr, "snd_pcm_hw_params: %s\n", snd_strerror(err));
        return err;
    }
```

---

### Sucesso

```c
    return 0;
}
```

Retorna zero para indicar sucesso.

---

## Função principal

```c
int main(int argc, char **argv) {
```

Ponto de entrada do programa.

* `argc`: número de argumentos.
* `argv`: vetor de strings com os argumentos.

---

### Defaults

```c
    const char *device = "hw:2,0";
```

Device ALSA padrão. Você pode trocar via linha de comando.

```c
    unsigned int rate = 48000;
```

Taxa de amostragem padrão.

```c
    snd_pcm_uframes_t frames_per_read = 256;
```

Quantidade de frames lidos por chamada.

Lembre:

* 1 frame = 1 amostra de todos os canais
* aqui, 1 frame = left + right

```c
    snd_pcm_uframes_t period_frames = 256;
```

Período ALSA padrão.

```c
    snd_pcm_uframes_t buffer_frames = 1024;
```

Buffer ALSA total padrão.

---

### Argumento 1: device

```c
    if (argc >= 2) {
        device = argv[1];
    }
```

Se o usuário passou um argumento, ele substitui o device padrão.

---

### Argumento 2: taxa

```c
    if (argc >= 3) {
        rate = (unsigned int)strtoul(argv[2], NULL, 10);
    }
```

Se passou um segundo argumento, converte para inteiro e usa como taxa.

---

### Argumento 3: frames por leitura

```c
    if (argc >= 4) {
        frames_per_read = (snd_pcm_uframes_t)strtoull(argv[3], NULL, 10);
        period_frames = frames_per_read;
        buffer_frames = frames_per_read * 4;
    }
```

Se passou um terceiro argumento:

* usa como número de frames por leitura,
* ajusta o período para o mesmo valor,
* e o buffer total para 4 vezes isso.

É uma forma simples de manter consistência entre leitura, período e buffer.

---

### Validação básica

```c
    if (rate == 0 || frames_per_read == 0) {
```

Evita valores inválidos.

```c
        fprintf(stderr, "Uso: %s [device] [rate] [frames_per_read]\n", argv[0]);
        return 1;
    }
```

Mostra uso e sai com erro.

---

### Configura sinais

```c
    setup_signals();
```

Agora `Ctrl+C` vai encerrar o loop com segurança.

---

## Abrir o dispositivo ALSA

```c
    snd_pcm_t *pcm = NULL;
```

Ponteiro para o handle do dispositivo ALSA.

```c
    int err = snd_pcm_open(&pcm, device, SND_PCM_STREAM_CAPTURE, 0);
```

Abre o device para captura.

* `&pcm`: onde guardar o handle.
* `device`: nome ALSA, tipo `hw:2,0`.
* `SND_PCM_STREAM_CAPTURE`: modo captura.
* `0`: sem flags especiais.

```c
    if (err < 0) {
        fprintf(stderr, "snd_pcm_open(%s): %s\n", device, snd_strerror(err));
        return 1;
    }
```

Se falhar, imprime erro e sai.

---

## Configurar hardware

```c
    if ((err = set_hw_params(pcm, rate, period_frames, buffer_frames)) < 0) {
        snd_pcm_close(pcm);
        return 1;
    }
```

Chama a função de configuração. Se falhar:

* fecha o device,
* sai com erro.

---

## Preparar stream

```c
    if ((err = snd_pcm_prepare(pcm)) < 0) {
```

Coloca o stream ALSA no estado preparado para começar a capturar.

```c
        fprintf(stderr, "snd_pcm_prepare: %s\n", snd_strerror(err));
        snd_pcm_close(pcm);
        return 1;
    }
```

---

## Alocar buffer de leitura

```c
    int32_t *buffer = calloc((size_t)frames_per_read * 2u, sizeof(int32_t));
```

Aloca memória para:

* `frames_per_read` frames
* 2 canais por frame
* cada canal `int32_t`

Exemplo:

* 256 frames
* 2 canais
* 512 inteiros de 32 bits

```c
    if (buffer == NULL) {
        fprintf(stderr, "Falha ao alocar buffer\n");
        snd_pcm_close(pcm);
        return 1;
    }
```

Se a alocação falhar, sai com erro.

---

## Mensagens iniciais

```c
    fprintf(stderr, "Capturando de %s, rate=%u, frames_per_read=%lu\n",
            device, rate, (unsigned long)frames_per_read);
```

Informa ao usuário o device, taxa e tamanho de leitura.

```c
    fprintf(stderr, "Formato: S32_LE, 2 canais, interleaved\n");
```

Explica o formato esperado.

```c
    fprintf(stderr, "Saída: LEFT RIGHT em hexadecimal\n");
```

Informa o formato da saída.

Tudo isso vai para `stderr`, deixando `stdout` reservado para os dados.

---

## Loop principal

```c
    while (!g_stop) {
```

Roda até o usuário mandar parar.

---

### Ler frames do ALSA

```c
        snd_pcm_sframes_t got = snd_pcm_readi(pcm, buffer, frames_per_read);
```

Lê até `frames_per_read` frames interleaved do device para `buffer`.

Retorna:

* número de frames realmente lidos, ou
* erro negativo.

---

### Tratar overrun

```c
        if (got == -EPIPE) {
```

`-EPIPE` em captura ALSA significa overrun.

```c
            fprintf(stderr, "Overrun detectado, rearmando stream...\n");
```

Avisa o usuário.

```c
            snd_pcm_prepare(pcm);
```

Reprepara o stream para voltar a funcionar.

```c
            continue;
        }
```

Volta ao início do loop.

---

### Tratar outros erros recuperáveis

```c
        if (got < 0) {
            got = snd_pcm_recover(pcm, (int)got, 1);
```

Tenta recuperar automaticamente o stream de outros erros ALSA.

```c
            if (got < 0) {
                fprintf(stderr, "snd_pcm_readi/recover: %s\n", snd_strerror((int)got));
                break;
            }
            continue;
        }
```

Se não conseguir recuperar:

* imprime erro,
* sai do loop.

Se recuperar:

* tenta de novo no próximo ciclo.

---

## Imprimir os words em hexadecimal

```c
        for (snd_pcm_sframes_t i = 0; i < got; ++i) {
```

Percorre todos os frames lidos.

---

### Canal esquerdo

```c
            uint32_t left = (uint32_t)buffer[(size_t)i * 2u + 0u];
```

Como o buffer é interleaved:

* índice do left do frame `i` = `2*i`
* índice do right do frame `i` = `2*i + 1`

Aqui pega o canal esquerdo e o reinterpreta como `uint32_t`.

Isso é importante porque:

* o dado original pode ser signed,
* mas para imprimir em hexadecimal queremos ver o padrão bruto de bits.

---

### Canal direito

```c
            uint32_t right = (uint32_t)buffer[(size_t)i * 2u + 1u];
```

Pega o canal direito do frame.

---

### Impressão

```c
            printf("0x%08" PRIX32 " 0x%08" PRIX32 "\n", left, right);
```

Imprime:

* `left` em hex com 8 dígitos
* `right` em hex com 8 dígitos

Exemplo:

```text
0x80012345 0x8006789A
```

`PRIX32` é uma macro portável para imprimir `uint32_t` em hexadecimal maiúsculo.

---

## Flush da saída

```c
        fflush(stdout);
```

Força a saída a aparecer imediatamente.

Útil quando você quer ver os valores em tempo real ou redirecionar para um arquivo sem atraso.

---

## Fim do loop

```c
    }
```

---

## Liberar memória

```c
    free(buffer);
```

Libera o buffer alocado.

---

## Fechar ALSA

```c
    snd_pcm_close(pcm);
```

Fecha o dispositivo ALSA.

---

## Sucesso

```c
    return 0;
}
```

Sai com código zero.

---

# Resumo conceitual do programa

Esse utilitário faz o mínimo necessário para depurar o canal I2S no Pi:

1. abre o device ALSA de captura,
2. configura:

   * `S32_LE`
   * 2 canais
   * taxa desejada
3. lê frames interleaved,
4. trata overrun,
5. imprime cada frame como:

```text
LEFT RIGHT
```

em hexadecimal bruto.

Então ele não faz:

* FFT,
* parser de tag,
* MFCC,
* comparação,

só mostra exatamente o que o ALSA entregou como words de 32 bits por canal.

# Leitura mental de um frame

Se sair:

```text
0x80012345 0x8006789A
```

você pode interpretar:

* word esquerdo = `0x80012345`
* word direito = `0x8006789A`

e, se o protocolo tagged tiver `tag_shift=30`, os 2 bits de tag estão no topo:

* `10` para ambos, por exemplo.

Aí você consegue ver se:

* as tags batem,
* left/right estão trocados,
* os bits reservados estão zerados,
* os words fazem sentido.

Se quiser, no próximo passo eu posso te dar uma **versão 2 desse código em C** que já imprime também:

* `tag`,
* `reserved`,
* `payload`,
* e detecta `tag_mismatch`.
