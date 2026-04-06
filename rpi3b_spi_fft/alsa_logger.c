#define _GNU_SOURCE

#include <alsa/asoundlib.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    CHANNEL_COUNT = 2,
    DEFAULT_RATE_HZ = 48828,
    DEFAULT_READ_FRAMES = 512,
    DEFAULT_PERIOD_FRAMES = 512,
    DEFAULT_BUFFER_FRAMES = 2048,
    DEFAULT_QUEUE_CHUNKS = 32,
    DEFAULT_FLUSH_EVERY_CHUNKS = 8,
    DEFAULT_STATS_INTERVAL_MS = 1000,
    DEFAULT_PIPE_SIZE_BYTES = 1 << 16,
};

typedef enum {
    OUTPUT_MODE_HEX = 0,
    OUTPUT_MODE_RAW = 1,
} output_mode_t;

typedef struct {
    const char *device;
    const char *output_path;
    unsigned int rate_hz;
    snd_pcm_uframes_t read_frames;
    snd_pcm_uframes_t period_frames;
    snd_pcm_uframes_t buffer_frames;
    size_t queue_chunks;
    unsigned int flush_every_chunks;
    unsigned int stats_interval_ms;
    int pipe_size_bytes;
    output_mode_t mode;
} capture_config_t;

typedef struct {
    int32_t *samples;
    snd_pcm_sframes_t *frame_counts;
    size_t slot_count;
    snd_pcm_uframes_t slot_frames;
    size_t read_index;
    size_t write_index;
    size_t used_slots;
    size_t high_water_slots;
    uint64_t full_waits;
    int closed;
    pthread_mutex_t mutex;
    pthread_cond_t can_read;
    pthread_cond_t can_write;
} capture_queue_t;

typedef struct {
    uint64_t captured_chunks;
    uint64_t captured_frames;
    uint64_t written_chunks;
    uint64_t written_frames;
    uint64_t xruns;
    uint64_t recoveries;
    uint64_t partial_reads;
} capture_stats_t;

typedef struct {
    capture_queue_t *queue;
    volatile sig_atomic_t *stop_flag;
    output_mode_t mode;
    int out_fd;
    FILE *hex_output;
    unsigned int flush_every_chunks;
    int close_fd_on_exit;
    int close_file_on_exit;
    int error_code;
    capture_stats_t *stats;
} writer_context_t;

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

static uint64_t monotonic_ms(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }

    return ((uint64_t)ts.tv_sec * 1000ULL) + ((uint64_t)ts.tv_nsec / 1000000ULL);
}

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Uso:\n"
            "  %s [device] [rate] [read_frames]\n"
            "  %s [opcoes]\n"
            "\n"
            "Opcoes:\n"
            "  -D, --device <hw:X,Y>        dispositivo ALSA (padrao: hw:2,0)\n"
            "  -r, --rate <Hz>              taxa de captura (padrao: 48828)\n"
            "  -n, --read-frames <N>        frames por leitura ALSA (padrao: 512)\n"
            "  -F, --period-frames <N>      tamanho do periodo ALSA (padrao: 512)\n"
            "  -B, --buffer-frames <N>      tamanho do buffer ALSA (padrao: 2048)\n"
            "  -Q, --queue-chunks <N>       chunks no buffer interno (padrao: 32)\n"
            "  -m, --mode <hex|raw>         formato da saida (padrao: hex)\n"
            "  -o, --output <path|- >       arquivo de saida ou stdout (padrao: -)\n"
            "      --pipe-size-bytes <N>    tamanho pedido para o pipe de stdout\n"
            "      --flush-every-chunks <N> flush do modo hex a cada N chunks\n"
            "      --stats-interval-ms <N>  intervalo dos contadores em stderr (0 desliga)\n"
            "  -h, --help                   mostra esta ajuda\n"
            "\n"
            "Modo raw escreve amostras S32_LE stereo diretamente, sem texto.\n",
            prog,
            prog);
}

static int parse_u32_arg(const char *text, unsigned int *out) {
    char *end = NULL;
    unsigned long value;

    if (text == NULL || *text == '\0') {
        return -1;
    }

    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value == 0 || value > 0xFFFFFFFFUL) {
        return -1;
    }

    *out = (unsigned int)value;
    return 0;
}

static int parse_nonneg_u32_arg(const char *text, unsigned int *out) {
    char *end = NULL;
    unsigned long value;

    if (text == NULL || *text == '\0') {
        return -1;
    }

    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > 0xFFFFFFFFUL) {
        return -1;
    }

    *out = (unsigned int)value;
    return 0;
}

static int parse_size_arg(const char *text, size_t *out) {
    char *end = NULL;
    unsigned long long value;

    if (text == NULL || *text == '\0') {
        return -1;
    }

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value == 0ULL) {
        return -1;
    }

    *out = (size_t)value;
    return 0;
}

static int parse_output_mode(const char *text, output_mode_t *mode) {
    if (text == NULL) {
        return -1;
    }
    if (strcmp(text, "hex") == 0) {
        *mode = OUTPUT_MODE_HEX;
        return 0;
    }
    if (strcmp(text, "raw") == 0) {
        *mode = OUTPUT_MODE_RAW;
        return 0;
    }
    return -1;
}

static int parse_args(int argc, char **argv, capture_config_t *cfg) {
    static const struct option long_opts[] = {
        {"device", required_argument, NULL, 'D'},
        {"rate", required_argument, NULL, 'r'},
        {"read-frames", required_argument, NULL, 'n'},
        {"period-frames", required_argument, NULL, 'F'},
        {"buffer-frames", required_argument, NULL, 'B'},
        {"queue-chunks", required_argument, NULL, 'Q'},
        {"mode", required_argument, NULL, 'm'},
        {"output", required_argument, NULL, 'o'},
        {"pipe-size-bytes", required_argument, NULL, 1000},
        {"flush-every-chunks", required_argument, NULL, 1001},
        {"stats-interval-ms", required_argument, NULL, 1002},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0},
    };

    unsigned int u32_value = 0;
    size_t size_value = 0;
    int opt = 0;
    int opt_index = 0;
    int period_explicit = 0;
    int buffer_explicit = 0;

    while ((opt = getopt_long(argc, argv, "D:r:n:F:B:Q:m:o:h", long_opts, &opt_index)) != -1) {
        switch (opt) {
            case 'D':
                cfg->device = optarg;
                break;
            case 'r':
                if (parse_u32_arg(optarg, &cfg->rate_hz) != 0) {
                    fprintf(stderr, "rate invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 'n':
                if (parse_u32_arg(optarg, &u32_value) != 0) {
                    fprintf(stderr, "read-frames invalido: %s\n", optarg);
                    return -1;
                }
                cfg->read_frames = (snd_pcm_uframes_t)u32_value;
                break;
            case 'F':
                if (parse_u32_arg(optarg, &u32_value) != 0) {
                    fprintf(stderr, "period-frames invalido: %s\n", optarg);
                    return -1;
                }
                cfg->period_frames = (snd_pcm_uframes_t)u32_value;
                period_explicit = 1;
                break;
            case 'B':
                if (parse_u32_arg(optarg, &u32_value) != 0) {
                    fprintf(stderr, "buffer-frames invalido: %s\n", optarg);
                    return -1;
                }
                cfg->buffer_frames = (snd_pcm_uframes_t)u32_value;
                buffer_explicit = 1;
                break;
            case 'Q':
                if (parse_size_arg(optarg, &size_value) != 0) {
                    fprintf(stderr, "queue-chunks invalido: %s\n", optarg);
                    return -1;
                }
                cfg->queue_chunks = size_value;
                break;
            case 'm':
                if (parse_output_mode(optarg, &cfg->mode) != 0) {
                    fprintf(stderr, "mode invalido: %s (use hex ou raw)\n", optarg);
                    return -1;
                }
                break;
            case 'o':
                cfg->output_path = optarg;
                break;
            case 1000:
                if (parse_nonneg_u32_arg(optarg, &u32_value) != 0) {
                    fprintf(stderr, "pipe-size-bytes invalido: %s\n", optarg);
                    return -1;
                }
                cfg->pipe_size_bytes = (int)u32_value;
                break;
            case 1001:
                if (parse_u32_arg(optarg, &cfg->flush_every_chunks) != 0) {
                    fprintf(stderr, "flush-every-chunks invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1002:
                if (parse_nonneg_u32_arg(optarg, &cfg->stats_interval_ms) != 0) {
                    fprintf(stderr, "stats-interval-ms invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 1;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    if (optind < argc) {
        cfg->device = argv[optind++];
    }
    if (optind < argc) {
        if (parse_u32_arg(argv[optind++], &cfg->rate_hz) != 0) {
            fprintf(stderr, "rate invalido: %s\n", argv[optind - 1]);
            return -1;
        }
    }
    if (optind < argc) {
        if (parse_u32_arg(argv[optind++], &u32_value) != 0) {
            fprintf(stderr, "read_frames invalido: %s\n", argv[optind - 1]);
            return -1;
        }
        cfg->read_frames = (snd_pcm_uframes_t)u32_value;
        if (!period_explicit) {
            cfg->period_frames = cfg->read_frames;
        }
        if (!buffer_explicit) {
            cfg->buffer_frames = cfg->read_frames * 4;
        }
    }
    if (optind < argc) {
        fprintf(stderr, "Argumentos extras inesperados a partir de: %s\n", argv[optind]);
        return -1;
    }

    if (cfg->buffer_frames < (cfg->period_frames * 2)) {
        cfg->buffer_frames = cfg->period_frames * 2;
    }

    return 0;
}

static int queue_init(capture_queue_t *queue, size_t slot_count, snd_pcm_uframes_t slot_frames) {
    size_t slot_samples = 0;
    size_t total_samples = 0;

    memset(queue, 0, sizeof(*queue));
    queue->slot_count = slot_count;
    queue->slot_frames = slot_frames;

    slot_samples = (size_t)slot_frames * CHANNEL_COUNT;
    if (slot_samples == 0 || slot_count == 0) {
        fprintf(stderr, "Fila interna invalida\n");
        return -1;
    }

    if (slot_samples > (SIZE_MAX / slot_count)) {
        fprintf(stderr, "Fila interna excede o limite de memoria\n");
        return -1;
    }

    total_samples = slot_samples * slot_count;
    queue->samples = calloc(total_samples, sizeof(int32_t));
    queue->frame_counts = calloc(slot_count, sizeof(snd_pcm_sframes_t));
    if (queue->samples == NULL || queue->frame_counts == NULL) {
        fprintf(stderr, "Falha ao alocar fila interna de captura\n");
        free(queue->samples);
        free(queue->frame_counts);
        memset(queue, 0, sizeof(*queue));
        return -1;
    }

    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->can_read, NULL);
    pthread_cond_init(&queue->can_write, NULL);
    return 0;
}

static void queue_close(capture_queue_t *queue) {
    pthread_mutex_lock(&queue->mutex);
    queue->closed = 1;
    pthread_cond_broadcast(&queue->can_read);
    pthread_cond_broadcast(&queue->can_write);
    pthread_mutex_unlock(&queue->mutex);
}

static void queue_destroy(capture_queue_t *queue) {
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->can_read);
    pthread_cond_destroy(&queue->can_write);
    free(queue->samples);
    free(queue->frame_counts);
    memset(queue, 0, sizeof(*queue));
}

static int32_t *queue_slot_ptr(capture_queue_t *queue, size_t slot_index) {
    size_t slot_samples = (size_t)queue->slot_frames * CHANNEL_COUNT;
    return &queue->samples[slot_index * slot_samples];
}

static int queue_acquire_write_slot(capture_queue_t *queue, size_t *slot_index) {
    pthread_mutex_lock(&queue->mutex);
    while (!g_stop && !queue->closed && (queue->used_slots == queue->slot_count)) {
        queue->full_waits += 1;
        pthread_cond_wait(&queue->can_write, &queue->mutex);
    }
    if (g_stop || queue->closed) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    *slot_index = queue->write_index;
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

static void queue_commit_write_slot(capture_queue_t *queue, size_t slot_index, snd_pcm_sframes_t frames) {
    pthread_mutex_lock(&queue->mutex);
    if (!queue->closed) {
        queue->frame_counts[slot_index] = frames;
        queue->write_index = (slot_index + 1) % queue->slot_count;
        queue->used_slots += 1;
        if (queue->used_slots > queue->high_water_slots) {
            queue->high_water_slots = queue->used_slots;
        }
        pthread_cond_signal(&queue->can_read);
    }
    pthread_mutex_unlock(&queue->mutex);
}

static int queue_acquire_read_slot(capture_queue_t *queue, size_t *slot_index, snd_pcm_sframes_t *frames) {
    pthread_mutex_lock(&queue->mutex);
    while ((queue->used_slots == 0) && !queue->closed) {
        pthread_cond_wait(&queue->can_read, &queue->mutex);
    }
    if (queue->used_slots == 0 && queue->closed) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    *slot_index = queue->read_index;
    *frames = queue->frame_counts[queue->read_index];
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

static void queue_release_read_slot(capture_queue_t *queue) {
    pthread_mutex_lock(&queue->mutex);
    if (queue->used_slots > 0) {
        queue->read_index = (queue->read_index + 1) % queue->slot_count;
        queue->used_slots -= 1;
        pthread_cond_signal(&queue->can_write);
    }
    pthread_mutex_unlock(&queue->mutex);
}

static void queue_snapshot(capture_queue_t *queue, size_t *used_slots, size_t *high_water_slots, uint64_t *full_waits) {
    pthread_mutex_lock(&queue->mutex);
    *used_slots = queue->used_slots;
    *high_water_slots = queue->high_water_slots;
    *full_waits = queue->full_waits;
    pthread_mutex_unlock(&queue->mutex);
}

static ssize_t write_all(int fd, const void *buffer, size_t bytes) {
    const uint8_t *ptr = (const uint8_t *)buffer;
    size_t offset = 0;

    while (offset < bytes) {
        ssize_t rc = write(fd, ptr + offset, bytes - offset);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (rc == 0) {
            errno = EIO;
            return -1;
        }
        offset += (size_t)rc;
    }

    return (ssize_t)offset;
}

static int write_hex_chunk(FILE *output, const int32_t *samples, snd_pcm_sframes_t frames) {
    snd_pcm_sframes_t idx;

    for (idx = 0; idx < frames; ++idx) {
        uint32_t left = (uint32_t)samples[(size_t)idx * CHANNEL_COUNT];
        uint32_t right = (uint32_t)samples[(size_t)idx * CHANNEL_COUNT + 1U];
        if (fprintf(output, "0x%08" PRIX32 " 0x%08" PRIX32 "\n", left, right) < 0) {
            return -1;
        }
    }

    return 0;
}

static int set_pipe_size_if_possible(int fd, int bytes) {
#ifdef F_SETPIPE_SZ
    if (bytes > 0 && fcntl(fd, F_SETPIPE_SZ, bytes) < 0) {
        return -1;
    }
#else
    (void)fd;
    (void)bytes;
#endif
    return 0;
}

static int open_output_sink(const capture_config_t *cfg, writer_context_t *writer) {
    writer->out_fd = -1;
    writer->hex_output = NULL;
    writer->close_fd_on_exit = 0;
    writer->close_file_on_exit = 0;

    if (cfg->mode == OUTPUT_MODE_RAW) {
        if (cfg->output_path == NULL || strcmp(cfg->output_path, "-") == 0) {
            if (isatty(STDOUT_FILENO)) {
                fprintf(stderr, "Modo raw nao pode escrever no terminal. Use --output arquivo.raw ou pipe.\n");
                return -1;
            }
            writer->out_fd = STDOUT_FILENO;
            if (cfg->pipe_size_bytes > 0) {
                (void)set_pipe_size_if_possible(writer->out_fd, cfg->pipe_size_bytes);
            }
            return 0;
        }

        writer->out_fd = open(cfg->output_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (writer->out_fd < 0) {
            fprintf(stderr, "Falha ao abrir %s: %s\n", cfg->output_path, strerror(errno));
            return -1;
        }
        writer->close_fd_on_exit = 1;
        return 0;
    }

    if (cfg->output_path == NULL || strcmp(cfg->output_path, "-") == 0) {
        writer->hex_output = stdout;
    } else {
        writer->hex_output = fopen(cfg->output_path, "w");
        if (writer->hex_output == NULL) {
            fprintf(stderr, "Falha ao abrir %s: %s\n", cfg->output_path, strerror(errno));
            return -1;
        }
        writer->close_file_on_exit = 1;
    }

    setvbuf(writer->hex_output, NULL, _IOFBF, 1 << 20);
    return 0;
}

static void close_output_sink(writer_context_t *writer) {
    if (writer->close_file_on_exit && writer->hex_output != NULL) {
        fclose(writer->hex_output);
    }
    if (writer->close_fd_on_exit && writer->out_fd >= 0) {
        close(writer->out_fd);
    }

    writer->hex_output = NULL;
    writer->out_fd = -1;
    writer->close_file_on_exit = 0;
    writer->close_fd_on_exit = 0;
}

static void *writer_main(void *opaque) {
    writer_context_t *writer = (writer_context_t *)opaque;

    while (1) {
        size_t slot_index = 0;
        snd_pcm_sframes_t frames = 0;
        int32_t *samples = NULL;

        if (queue_acquire_read_slot(writer->queue, &slot_index, &frames) != 0) {
            break;
        }

        samples = queue_slot_ptr(writer->queue, slot_index);
        if (writer->mode == OUTPUT_MODE_RAW) {
            size_t chunk_bytes = (size_t)frames * CHANNEL_COUNT * sizeof(int32_t);
            if (write_all(writer->out_fd, samples, chunk_bytes) < 0) {
                writer->error_code = errno != 0 ? errno : EIO;
                *writer->stop_flag = 1;
                queue_close(writer->queue);
                break;
            }
        } else {
            if (write_hex_chunk(writer->hex_output, samples, frames) != 0) {
                writer->error_code = errno != 0 ? errno : EIO;
                *writer->stop_flag = 1;
                queue_close(writer->queue);
                break;
            }
            if (writer->flush_every_chunks > 0 &&
                (((writer->stats->written_chunks + 1ULL) % writer->flush_every_chunks) == 0ULL)) {
                fflush(writer->hex_output);
            }
        }

        writer->stats->written_chunks += 1;
        writer->stats->written_frames += (uint64_t)frames;
        queue_release_read_slot(writer->queue);
    }

    if (writer->mode == OUTPUT_MODE_HEX && writer->hex_output != NULL) {
        fflush(writer->hex_output);
    }
    return NULL;
}

static int set_hw_params(
    snd_pcm_t *pcm,
    unsigned int *rate_hz,
    snd_pcm_uframes_t *period_frames,
    snd_pcm_uframes_t *buffer_frames
) {
    int err = 0;
    int dir = 0;
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
    if ((err = snd_pcm_hw_params_set_channels(pcm, hw, CHANNEL_COUNT)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_channels: %s\n", snd_strerror(err));
        return err;
    }
    if ((err = snd_pcm_hw_params_set_rate_near(pcm, hw, rate_hz, &dir)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_rate_near: %s\n", snd_strerror(err));
        return err;
    }
    dir = 0;
    if ((err = snd_pcm_hw_params_set_period_size_near(pcm, hw, period_frames, &dir)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_period_size_near: %s\n", snd_strerror(err));
        return err;
    }
    if ((err = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, buffer_frames)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params_set_buffer_size_near: %s\n", snd_strerror(err));
        return err;
    }
    if ((err = snd_pcm_hw_params(pcm, hw)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params: %s\n", snd_strerror(err));
        return err;
    }

    snd_pcm_hw_params_get_rate(hw, rate_hz, &dir);
    snd_pcm_hw_params_get_period_size(hw, period_frames, &dir);
    snd_pcm_hw_params_get_buffer_size(hw, buffer_frames);
    return 0;
}

static int set_sw_params(snd_pcm_t *pcm, snd_pcm_uframes_t avail_min, snd_pcm_uframes_t buffer_frames) {
    int err = 0;
    snd_pcm_sw_params_t *sw = NULL;

    snd_pcm_sw_params_alloca(&sw);
    if ((err = snd_pcm_sw_params_current(pcm, sw)) < 0) {
        fprintf(stderr, "snd_pcm_sw_params_current: %s\n", snd_strerror(err));
        return err;
    }
    if ((err = snd_pcm_sw_params_set_avail_min(pcm, sw, avail_min)) < 0) {
        fprintf(stderr, "snd_pcm_sw_params_set_avail_min: %s\n", snd_strerror(err));
        return err;
    }
    if ((err = snd_pcm_sw_params_set_start_threshold(pcm, sw, 1)) < 0) {
        fprintf(stderr, "snd_pcm_sw_params_set_start_threshold: %s\n", snd_strerror(err));
        return err;
    }
    if ((err = snd_pcm_sw_params_set_stop_threshold(pcm, sw, buffer_frames)) < 0) {
        fprintf(stderr, "snd_pcm_sw_params_set_stop_threshold: %s\n", snd_strerror(err));
        return err;
    }
    if ((err = snd_pcm_sw_params(pcm, sw)) < 0) {
        fprintf(stderr, "snd_pcm_sw_params: %s\n", snd_strerror(err));
        return err;
    }
    return 0;
}

static int recover_capture_error(snd_pcm_t *pcm, int err, capture_stats_t *stats, const char *context) {
    int recovered;

    if (err == -EPIPE) {
        stats->xruns += 1;
    }

    recovered = snd_pcm_recover(pcm, err, 1);
    if (recovered < 0) {
        fprintf(stderr, "%s: %s\n", context, snd_strerror(recovered));
        return recovered;
    }

    stats->recoveries += 1;
    return 0;
}

static void print_stats(
    const capture_config_t *cfg,
    capture_queue_t *queue,
    const capture_stats_t *stats,
    bool final_report
) {
    size_t used_slots = 0;
    size_t high_water_slots = 0;
    uint64_t full_waits = 0;
    const char *label = final_report ? "final" : "stats";

    queue_snapshot(queue, &used_slots, &high_water_slots, &full_waits);

    fprintf(stderr,
            "[%s] mode=%s rate=%u captured_chunks=%" PRIu64 " captured_frames=%" PRIu64
            " written_chunks=%" PRIu64 " written_frames=%" PRIu64
            " xruns=%" PRIu64 " recoveries=%" PRIu64 " partial_reads=%" PRIu64
            " queue=%zu/%zu queue_high=%zu full_waits=%" PRIu64 "\n",
            label,
            cfg->mode == OUTPUT_MODE_RAW ? "raw" : "hex",
            cfg->rate_hz,
            stats->captured_chunks,
            stats->captured_frames,
            stats->written_chunks,
            stats->written_frames,
            stats->xruns,
            stats->recoveries,
            stats->partial_reads,
            used_slots,
            queue->slot_count,
            high_water_slots,
            full_waits);
}

int main(int argc, char **argv) {
    capture_config_t cfg = {
        .device = "hw:2,0",
        .output_path = "-",
        .rate_hz = DEFAULT_RATE_HZ,
        .read_frames = DEFAULT_READ_FRAMES,
        .period_frames = DEFAULT_PERIOD_FRAMES,
        .buffer_frames = DEFAULT_BUFFER_FRAMES,
        .queue_chunks = DEFAULT_QUEUE_CHUNKS,
        .flush_every_chunks = DEFAULT_FLUSH_EVERY_CHUNKS,
        .stats_interval_ms = DEFAULT_STATS_INTERVAL_MS,
        .pipe_size_bytes = DEFAULT_PIPE_SIZE_BYTES,
        .mode = OUTPUT_MODE_HEX,
    };
    capture_queue_t queue;
    capture_stats_t stats;
    writer_context_t writer;
    snd_pcm_t *pcm = NULL;
    pthread_t writer_thread;
    int writer_started = 0;
    int exit_code = 1;
    int err = 0;
    uint64_t last_stats_ms = 0;

    memset(&queue, 0, sizeof(queue));
    memset(&stats, 0, sizeof(stats));
    memset(&writer, 0, sizeof(writer));

    err = parse_args(argc, argv, &cfg);
    if (err > 0) {
        return 0;
    }
    if (err < 0) {
        return 1;
    }

    setup_signals();

    if (queue_init(&queue, cfg.queue_chunks, cfg.read_frames) != 0) {
        return 1;
    }

    writer.queue = &queue;
    writer.stop_flag = &g_stop;
    writer.mode = cfg.mode;
    writer.flush_every_chunks = cfg.flush_every_chunks;
    writer.stats = &stats;

    if (open_output_sink(&cfg, &writer) != 0) {
        queue_destroy(&queue);
        return 1;
    }

    if (pthread_create(&writer_thread, NULL, writer_main, &writer) != 0) {
        fprintf(stderr, "Falha ao iniciar thread de escrita\n");
        close_output_sink(&writer);
        queue_destroy(&queue);
        return 1;
    }
    writer_started = 1;

    err = snd_pcm_open(&pcm, cfg.device, SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        fprintf(stderr, "snd_pcm_open(%s): %s\n", cfg.device, snd_strerror(err));
        goto cleanup;
    }

    if ((err = set_hw_params(pcm, &cfg.rate_hz, &cfg.period_frames, &cfg.buffer_frames)) < 0) {
        goto cleanup;
    }
    if ((err = set_sw_params(pcm, cfg.period_frames, cfg.buffer_frames)) < 0) {
        goto cleanup;
    }
    if ((err = snd_pcm_prepare(pcm)) < 0) {
        fprintf(stderr, "snd_pcm_prepare: %s\n", snd_strerror(err));
        goto cleanup;
    }

    fprintf(stderr,
            "Captura ALSA iniciada: device=%s rate=%u read_frames=%lu period=%lu buffer=%lu queue_chunks=%zu mode=%s\n",
            cfg.device,
            cfg.rate_hz,
            (unsigned long)cfg.read_frames,
            (unsigned long)cfg.period_frames,
            (unsigned long)cfg.buffer_frames,
            cfg.queue_chunks,
            cfg.mode == OUTPUT_MODE_RAW ? "raw" : "hex");

    last_stats_ms = monotonic_ms();

    while (!g_stop) {
        size_t slot_index = 0;
        snd_pcm_sframes_t got = 0;
        int32_t *slot = NULL;

        if (queue_acquire_write_slot(&queue, &slot_index) != 0) {
            g_stop = 1;
            break;
        }

        slot = queue_slot_ptr(&queue, slot_index);
        got = snd_pcm_readi(pcm, slot, cfg.read_frames);
        if (got < 0) {
            if (recover_capture_error(pcm, (int)got, &stats, "snd_pcm_readi") < 0) {
                goto cleanup;
            }
            continue;
        }
        if (got == 0) {
            continue;
        }
        if ((snd_pcm_uframes_t)got < cfg.read_frames) {
            stats.partial_reads += 1;
        }

        queue_commit_write_slot(&queue, slot_index, got);
        stats.captured_chunks += 1;
        stats.captured_frames += (uint64_t)got;
        if (cfg.stats_interval_ms > 0 && (monotonic_ms() - last_stats_ms) >= cfg.stats_interval_ms) {
            print_stats(&cfg, &queue, &stats, false);
            last_stats_ms = monotonic_ms();
        }
    }

    exit_code = 0;

cleanup:
    queue_close(&queue);
    if (writer_started) {
        pthread_join(writer_thread, NULL);
    }
    if (pcm != NULL) {
        snd_pcm_close(pcm);
        pcm = NULL;
    }

    if (writer.error_code != 0) {
        fprintf(stderr, "Falha ao escrever a saida: %s\n", strerror(writer.error_code));
        exit_code = 1;
    }

    print_stats(&cfg, &queue, &stats, true);
    close_output_sink(&writer);
    queue_destroy(&queue);
    return exit_code;
}
