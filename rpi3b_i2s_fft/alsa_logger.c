#include <alsa/asoundlib.h>
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t g_stop = 0;

static void handle_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static void setup_signals(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}

static int set_hw_params(
    snd_pcm_t *pcm,
    unsigned int rate,
    snd_pcm_uframes_t period_frames,
    snd_pcm_uframes_t buffer_frames
) {
    int err;
    snd_pcm_hw_params_t *hw = NULL;

    snd_pcm_hw_params_alloca(&hw);

    if ((err = snd_pcm_hw_params_any(pcm, hw)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_any: %s\n", snd_strerror(err));
        return err;
    }

    if ((err = snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_access: %s\n", snd_strerror(err));
        return err;
    }

    if ((err = snd_pcm_hw_params_set_format(pcm, hw, SND_PCM_FORMAT_S32_LE)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_format: %s\n", snd_strerror(err));
        return err;
    }

    if ((err = snd_pcm_hw_params_set_channels(pcm, hw, 2)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_channels: %s\n", snd_strerror(err));
        return err;
    }

    if ((err = snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, 0)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_rate_near: %s\n", snd_strerror(err));
        return err;
    }

    if ((err = snd_pcm_hw_params_set_period_size_near(pcm, hw, &period_frames, 0)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_period_size_near: %s\n", snd_strerror(err));
        return err;
    }

    if ((err = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &buffer_frames)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_buffer_size_near: %s\n", snd_strerror(err));
        return err;
    }

    if ((err = snd_pcm_hw_params(pcm, hw)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params: %s\n", snd_strerror(err));
        return err;
    }

    return 0;
}

int main(int argc, char **argv) {
    const char *device = "hw:2,0";
    unsigned int rate = 48828;
    snd_pcm_uframes_t frames_per_read = 512;
    snd_pcm_uframes_t period_frames = 512;
    snd_pcm_uframes_t buffer_frames = 2048;

    if (argc >= 2) {
        device = argv[1];
    }
    if (argc >= 3) {
        rate = (unsigned int)strtoul(argv[2], NULL, 10);
    }
    if (argc >= 4) {
        frames_per_read = (snd_pcm_uframes_t)strtoull(argv[3], NULL, 10);
        period_frames = frames_per_read;
        buffer_frames = frames_per_read * 4;
    }

    if (rate == 0 || frames_per_read == 0) {
        fprintf(stderr, "Uso: %s [device] [rate] [frames_per_read]\n", argv[0]);
        fprintf(stderr, "Rate padrao do host: 48828 Hz (wire nominal = 48828.125 Hz)\n");
        return 1;
    }

    setup_signals();

    snd_pcm_t *pcm = NULL;
    int err = snd_pcm_open(&pcm, device, SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        fprintf(stderr, "snd_pcm_open(%s): %s\n", device, snd_strerror(err));
        return 1;
    }

    if ((err = set_hw_params(pcm, rate, period_frames, buffer_frames)) < 0) {
        snd_pcm_close(pcm);
        return 1;
    }

    if ((err = snd_pcm_prepare(pcm)) < 0) {
        fprintf(stderr, "snd_pcm_prepare: %s\n", snd_strerror(err));
        snd_pcm_close(pcm);
        return 1;
    }

    int32_t *buffer = calloc((size_t)frames_per_read * 2u, sizeof(int32_t));
    if (buffer == NULL) {
        fprintf(stderr, "Falha ao alocar buffer\n");
        snd_pcm_close(pcm);
        return 1;
    }

    fprintf(stderr, "Capturando de %s, rate=%u, frames_per_read=%lu\n",
            device, rate, (unsigned long)frames_per_read);
    fprintf(stderr, "Formato: S32_LE, 2 canais, interleaved\n");
    fprintf(stderr, "Saída: LEFT RIGHT em hexadecimal\n");

    while (!g_stop) {
        snd_pcm_sframes_t got = snd_pcm_readi(pcm, buffer, frames_per_read);

        if (got == -EPIPE) {
            fprintf(stderr, "Overrun detectado, rearmando stream...\n");
            snd_pcm_prepare(pcm);
            continue;
        }

        if (got < 0) {
            got = snd_pcm_recover(pcm, (int)got, 1);
            if (got < 0) {
                fprintf(stderr, "snd_pcm_readi/recover: %s\n", snd_strerror((int)got));
                break;
            }
            continue;
        }

        for (snd_pcm_sframes_t i = 0; i < got; ++i) {
            uint32_t left = (uint32_t)buffer[(size_t)i * 2u + 0u];
            uint32_t right = (uint32_t)buffer[(size_t)i * 2u + 1u];
            printf("0x%08" PRIX32 " 0x%08" PRIX32 "\n", left, right);
        }

        fflush(stdout);
    }

    free(buffer);
    snd_pcm_close(pcm);
    return 0;
}
