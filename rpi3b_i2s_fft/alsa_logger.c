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
    DEFAULT_PACKET_INDEX_BITS = 10,
    DEFAULT_PACKET_INDEX_SHIFT = 22,
    DEFAULT_FFT_PACKET_INDEX_BASE = 1 << (DEFAULT_PACKET_INDEX_BITS - 1),
    DEFAULT_TAG_SHIFT = 20,
    DEFAULT_TAG_MASK = 0x3,
    DEFAULT_PAYLOAD_BITS = 18,
    DEFAULT_TAG_IDLE = 0,
    DEFAULT_TAG_BFPEXP = 1,
    DEFAULT_TAG_FFT = 2,
};

typedef enum {
    OUTPUT_MODE_HEX = 0,
    OUTPUT_MODE_RAW = 1,
    OUTPUT_MODE_FFT_RAW = 2,
} output_mode_t;

typedef enum {
    TELEMETRY_FORMAT_TEXT = 0,
    TELEMETRY_FORMAT_JSONL = 1,
} telemetry_format_t;

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
    telemetry_format_t telemetry_format;
    unsigned int realign_initial_word_skip;
    int realign_swap_channels;
    int show_tagged_fields;
    unsigned int packet_index_shift;
    unsigned int packet_index_bits;
    unsigned int fft_packet_index_base;
    unsigned int tag_shift;
    unsigned int tag_mask;
    unsigned int payload_bits;
    unsigned int tag_idle;
    unsigned int tag_bfpexp;
    unsigned int tag_fft;
    unsigned int fft_frame_bins;
    unsigned int bfpexp_hold_pairs;
    unsigned int loss_tolerance_pairs;
    int allow_fft_without_bfpexp;
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
    size_t initial_word_skip_remaining;
    int have_pending_word;
    int32_t pending_word;
} realign_state_t;

typedef struct {
    int waiting_for_start;
    int current_bfpexp;
    int have_explicit_bfpexp;
    unsigned int bfpexp_gap_count;
    unsigned int bfpexp_seen_count;
    unsigned int highest_bin_index_seen;
    unsigned int received_count;
    int32_t *frame_pairs;
    uint8_t *received_bins;
    uint8_t *bfpexp_seen;
} tagged_fft_state_t;

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
    const capture_config_t *cfg;
    realign_state_t realign_state;
    tagged_fft_state_t tagged_fft_state;
    int32_t *transform_buffer;
    size_t transform_capacity_words;
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

static uint64_t realtime_ns(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        return 0;
    }

    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Uso:\n"
            "  %s [device] [rate] [read_frames]\n"
            "  %s [opcoes]\n"
            "\n"
            "Opcoes:\n"
            "  -D, --device <hw:X,Y>        dispositivo ALSA (padrao: hw:2,0)\n"
            "  -r, --rate <Hz>              taxa nominal de captura (padrao: 48828)\n"
            "  -n, --read-frames <N>        frames por leitura ALSA (padrao: 512)\n"
            "  -F, --period-frames <N>      tamanho do periodo ALSA (padrao: 512)\n"
            "  -B, --buffer-frames <N>      tamanho do buffer ALSA (padrao: 2048)\n"
            "  -Q, --queue-chunks <N>       chunks no buffer interno (padrao: 32)\n"
            "  -m, --mode <hex|raw|fft-raw> formato da saida (padrao: hex)\n"
            "  -o, --output <path|- >       arquivo de saida ou stdout (padrao: -)\n"
            "      --telemetry-format <fmt> stderr em text ou jsonl (padrao: text)\n"
            "      --realign-tagged         preset observado no Pi: descarta 1 word inicial e troca L/R\n"
            "      --realign-initial-word-skip <N>\n"
            "                               descarta N palavras S32 antes de reemparelhar o stream\n"
            "      --realign-swap-channels  troca left/right apos o reemparelhamento\n"
            "      --show-tagged-fields     acrescenta dec, kind, packet_index, tag e bin no modo hex\n"
            "      --packet-index-shift <N> shift do campo packet_index (padrao: 22)\n"
            "      --packet-index-bits <N>  largura do campo packet_index (padrao: 10)\n"
            "      --fft-packet-index-base <N>\n"
            "                               primeiro packet_index usado pelos bins FFT (padrao: 512)\n"
            "      --tag-shift <N>          shift do campo tag (padrao: 20)\n"
            "      --tag-mask <N>           mascara do campo tag (padrao: 0x3)\n"
            "      --payload-bits <N>       largura do payload assinado em cada word (padrao: 18)\n"
            "      --tag-idle <N>           valor da tag idle (padrao: 0)\n"
            "      --tag-bfpexp <N>         valor da tag BFPEXP (padrao: 1)\n"
            "      --tag-fft <N>            valor da tag FFT (padrao: 2)\n"
            "      --fft-frame-bins <N>     bins por quadro FFT emitido em fft-raw (padrao: 512)\n"
            "      --bfpexp-hold-pairs <N>  BFPEXP consecutivos requeridos antes do burst FFT (padrao: 128)\n"
            "      --loss-tolerance-pairs <N>\n"
            "                               perdas/corrupcoes toleradas por preambulo/burst (padrao: 3)\n"
            "      --allow-fft-without-bfpexp\n"
            "                               aceita inicio de burst FFT sem preambulo BFPEXP completo\n"
            "      --pipe-size-bytes <N>    tamanho pedido para o pipe de stdout\n"
            "      --flush-every-chunks <N> flush do modo hex a cada N chunks\n"
            "      --stats-interval-ms <N>  intervalo dos contadores em stderr (0 desliga)\n"
            "  -h, --help                   mostra esta ajuda\n"
            "\n"
            "Modo raw escreve amostras S32_LE stereo diretamente, sem texto.\n"
            "Modo fft-raw escreve quadros FFT decodificados em binario: cabecalho de 4 int32\n"
            "(bfpexp, flags, received_bins, reservado) seguido de frame_bins pares real/imag S32_LE.\n"
            "Em modo I2S slave, --rate ajusta a taxa nominal pedida ao ALSA; o clock\n"
            "fisico continua vindo da FPGA. Com --realign-tagged/--show-tagged-fields,\n"
            "o modo hex tambem exibe payload decimal do par, kind, packet_index, tag e bin FFT decodificados.\n",
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
    if (strcmp(text, "fft-raw") == 0) {
        *mode = OUTPUT_MODE_FFT_RAW;
        return 0;
    }
    return -1;
}

static int parse_telemetry_format(const char *text, telemetry_format_t *format) {
    if (text == NULL) {
        return -1;
    }
    if (strcmp(text, "text") == 0) {
        *format = TELEMETRY_FORMAT_TEXT;
        return 0;
    }
    if (strcmp(text, "jsonl") == 0) {
        *format = TELEMETRY_FORMAT_JSONL;
        return 0;
    }
    return -1;
}

static size_t tag_mask_bit_width(uint32_t mask) {
    size_t width = 0;

    while (mask != 0U) {
        width += 1U;
        mask >>= 1U;
    }

    return width;
}

static int realignment_enabled(const capture_config_t *cfg) {
    return cfg->realign_initial_word_skip > 0U || cfg->realign_swap_channels;
}

static const char *output_mode_name(output_mode_t mode) {
    if (mode == OUTPUT_MODE_RAW) {
        return "raw";
    }
    if (mode == OUTPUT_MODE_FFT_RAW) {
        return "fft-raw";
    }
    return "hex";
}

static void queue_snapshot(capture_queue_t *queue, size_t *used_slots, size_t *high_water_slots, uint64_t *full_waits);

static void json_write_escaped(FILE *stream, const char *text) {
    const unsigned char *ptr = (const unsigned char *)(text != NULL ? text : "");

    fputc('"', stream);
    while (*ptr != '\0') {
        unsigned char ch = *ptr++;
        switch (ch) {
            case '\\':
            case '"':
                fputc('\\', stream);
                fputc((int)ch, stream);
                break;
            case '\b':
                fputs("\\b", stream);
                break;
            case '\f':
                fputs("\\f", stream);
                break;
            case '\n':
                fputs("\\n", stream);
                break;
            case '\r':
                fputs("\\r", stream);
                break;
            case '\t':
                fputs("\\t", stream);
                break;
            default:
                if (ch < 0x20U) {
                    fprintf(stream, "\\u%04X", (unsigned int)ch);
                } else {
                    fputc((int)ch, stream);
                }
                break;
        }
    }
    fputc('"', stream);
}

static void telemetry_write_string_field(FILE *stream, const char *key, const char *value) {
    fprintf(stream, ",\"%s\":", key);
    json_write_escaped(stream, value);
}

static void telemetry_write_u64_field(FILE *stream, const char *key, uint64_t value) {
    fprintf(stream, ",\"%s\":%" PRIu64, key, value);
}

static void telemetry_write_i64_field(FILE *stream, const char *key, int64_t value) {
    fprintf(stream, ",\"%s\":%" PRId64, key, value);
}

static void telemetry_write_bool_field(FILE *stream, const char *key, bool value) {
    fprintf(stream, ",\"%s\":%s", key, value ? "true" : "false");
}

static void telemetry_write_queue_fields(FILE *stream, capture_queue_t *queue) {
    size_t used_slots = 0;
    size_t high_water_slots = 0;
    uint64_t full_waits = 0;

    queue_snapshot(queue, &used_slots, &high_water_slots, &full_waits);
    telemetry_write_u64_field(stream, "queue_used_slots", (uint64_t)used_slots);
    telemetry_write_u64_field(stream, "queue_high_water_slots", (uint64_t)high_water_slots);
    telemetry_write_u64_field(stream, "queue_full_waits", full_waits);
    telemetry_write_u64_field(stream, "queue_slot_count", (uint64_t)queue->slot_count);
}

static void emit_capture_session_start(
    const capture_config_t *cfg,
    capture_queue_t *queue,
    unsigned int requested_rate_hz
) {
    if (cfg->telemetry_format == TELEMETRY_FORMAT_TEXT) {
        fprintf(stderr,
                "Captura ALSA iniciada: device=%s requested_rate=%u actual_rate=%u read_frames=%lu period=%lu buffer=%lu queue_chunks=%zu mode=%s realign_word_skip=%u realign_swap=%s tagged_fields=%s\n",
                cfg->device,
                requested_rate_hz,
                cfg->rate_hz,
                (unsigned long)cfg->read_frames,
                (unsigned long)cfg->period_frames,
                (unsigned long)cfg->buffer_frames,
                cfg->queue_chunks,
                output_mode_name(cfg->mode),
                cfg->realign_initial_word_skip,
                cfg->realign_swap_channels ? "yes" : "no",
                cfg->show_tagged_fields ? "yes" : "no");
        return;
    }

    fprintf(stderr, "{\"type\":\"capture_session_start\"");
    telemetry_write_u64_field(stderr, "timestamp_ns", realtime_ns());
    telemetry_write_string_field(stderr, "device", cfg->device);
    telemetry_write_u64_field(stderr, "requested_rate_hz", (uint64_t)requested_rate_hz);
    telemetry_write_u64_field(stderr, "actual_rate_hz", (uint64_t)cfg->rate_hz);
    telemetry_write_u64_field(stderr, "read_frames", (uint64_t)cfg->read_frames);
    telemetry_write_u64_field(stderr, "period_frames", (uint64_t)cfg->period_frames);
    telemetry_write_u64_field(stderr, "buffer_frames", (uint64_t)cfg->buffer_frames);
    telemetry_write_u64_field(stderr, "queue_chunks", (uint64_t)cfg->queue_chunks);
    telemetry_write_u64_field(stderr, "flush_every_chunks", (uint64_t)cfg->flush_every_chunks);
    telemetry_write_u64_field(stderr, "stats_interval_ms", (uint64_t)cfg->stats_interval_ms);
    telemetry_write_u64_field(stderr, "pipe_size_bytes", (uint64_t)(cfg->pipe_size_bytes >= 0 ? cfg->pipe_size_bytes : 0));
    telemetry_write_string_field(stderr, "mode", output_mode_name(cfg->mode));
    telemetry_write_string_field(stderr, "telemetry_format", "jsonl");
    telemetry_write_u64_field(stderr, "realign_initial_word_skip", (uint64_t)cfg->realign_initial_word_skip);
    telemetry_write_bool_field(stderr, "realign_swap_channels", cfg->realign_swap_channels != 0);
    telemetry_write_bool_field(stderr, "show_tagged_fields", cfg->show_tagged_fields != 0);
    telemetry_write_u64_field(stderr, "packet_index_shift", (uint64_t)cfg->packet_index_shift);
    telemetry_write_u64_field(stderr, "packet_index_bits", (uint64_t)cfg->packet_index_bits);
    telemetry_write_u64_field(stderr, "fft_packet_index_base", (uint64_t)cfg->fft_packet_index_base);
    telemetry_write_u64_field(stderr, "tag_shift", (uint64_t)cfg->tag_shift);
    telemetry_write_u64_field(stderr, "tag_mask", (uint64_t)cfg->tag_mask);
    telemetry_write_u64_field(stderr, "tag_idle", (uint64_t)cfg->tag_idle);
    telemetry_write_u64_field(stderr, "tag_bfpexp", (uint64_t)cfg->tag_bfpexp);
    telemetry_write_u64_field(stderr, "tag_fft", (uint64_t)cfg->tag_fft);
    telemetry_write_queue_fields(stderr, queue);
    fputs("}\n", stderr);
    fflush(stderr);
}

static void emit_capture_warning_rate_adjusted(
    const capture_config_t *cfg,
    unsigned int requested_rate_hz
) {
    if (cfg->telemetry_format == TELEMETRY_FORMAT_TEXT) {
        fprintf(stderr,
                "Aviso: ALSA negociou rate=%u Hz apos pedido de %u Hz\n",
                cfg->rate_hz,
                requested_rate_hz);
        return;
    }

    fprintf(stderr, "{\"type\":\"capture_warning\"");
    telemetry_write_u64_field(stderr, "timestamp_ns", realtime_ns());
    telemetry_write_string_field(stderr, "warning", "rate_adjusted");
    telemetry_write_u64_field(stderr, "requested_rate_hz", (uint64_t)requested_rate_hz);
    telemetry_write_u64_field(stderr, "actual_rate_hz", (uint64_t)cfg->rate_hz);
    fputs("}\n", stderr);
    fflush(stderr);
}

static void emit_capture_chunk(
    const capture_config_t *cfg,
    capture_queue_t *queue,
    const capture_stats_t *stats,
    snd_pcm_sframes_t frames,
    bool partial_read
) {
    if (cfg->telemetry_format != TELEMETRY_FORMAT_JSONL) {
        return;
    }

    fprintf(stderr, "{\"type\":\"capture_chunk\"");
    telemetry_write_u64_field(stderr, "timestamp_ns", realtime_ns());
    telemetry_write_u64_field(stderr, "chunk_index", stats->captured_chunks > 0 ? stats->captured_chunks - 1ULL : 0ULL);
    telemetry_write_u64_field(stderr, "frames", (uint64_t)frames);
    telemetry_write_bool_field(stderr, "partial_read", partial_read);
    telemetry_write_u64_field(stderr, "captured_chunks", stats->captured_chunks);
    telemetry_write_u64_field(stderr, "captured_frames", stats->captured_frames);
    telemetry_write_u64_field(stderr, "written_chunks", stats->written_chunks);
    telemetry_write_u64_field(stderr, "written_frames", stats->written_frames);
    telemetry_write_u64_field(stderr, "xruns", stats->xruns);
    telemetry_write_u64_field(stderr, "recoveries", stats->recoveries);
    telemetry_write_u64_field(stderr, "partial_reads", stats->partial_reads);
    telemetry_write_queue_fields(stderr, queue);
    fputs("}\n", stderr);
    fflush(stderr);
}

static void emit_capture_recovery(
    const capture_config_t *cfg,
    capture_queue_t *queue,
    const capture_stats_t *stats,
    const char *context,
    int err,
    int recovered
) {
    if (cfg->telemetry_format != TELEMETRY_FORMAT_JSONL) {
        return;
    }

    fprintf(stderr, "{\"type\":\"capture_recovery\"");
    telemetry_write_u64_field(stderr, "timestamp_ns", realtime_ns());
    telemetry_write_string_field(stderr, "context", context);
    telemetry_write_i64_field(stderr, "error_code", (int64_t)err);
    telemetry_write_string_field(stderr, "error_name", snd_strerror(err));
    telemetry_write_i64_field(stderr, "recover_result", (int64_t)recovered);
    telemetry_write_bool_field(stderr, "xrun", err == -EPIPE);
    telemetry_write_u64_field(stderr, "xruns", stats->xruns);
    telemetry_write_u64_field(stderr, "recoveries", stats->recoveries);
    telemetry_write_u64_field(stderr, "partial_reads", stats->partial_reads);
    telemetry_write_queue_fields(stderr, queue);
    fputs("}\n", stderr);
    fflush(stderr);
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
        {"telemetry-format", required_argument, NULL, 999},
        {"pipe-size-bytes", required_argument, NULL, 1000},
        {"flush-every-chunks", required_argument, NULL, 1001},
        {"stats-interval-ms", required_argument, NULL, 1002},
        {"realign-tagged", no_argument, NULL, 1003},
        {"realign-initial-word-skip", required_argument, NULL, 1004},
        {"realign-swap-channels", no_argument, NULL, 1005},
        {"show-tagged-fields", no_argument, NULL, 1006},
        {"packet-index-shift", required_argument, NULL, 1007},
        {"packet-index-bits", required_argument, NULL, 1008},
        {"fft-packet-index-base", required_argument, NULL, 1009},
        {"tag-shift", required_argument, NULL, 1010},
        {"tag-mask", required_argument, NULL, 1011},
        {"payload-bits", required_argument, NULL, 1012},
        {"tag-idle", required_argument, NULL, 1013},
        {"tag-bfpexp", required_argument, NULL, 1014},
        {"tag-fft", required_argument, NULL, 1015},
        {"fft-frame-bins", required_argument, NULL, 1016},
        {"bfpexp-hold-pairs", required_argument, NULL, 1017},
        {"loss-tolerance-pairs", required_argument, NULL, 1018},
        {"allow-fft-without-bfpexp", no_argument, NULL, 1019},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0},
    };

    unsigned int u32_value = 0;
    size_t size_value = 0;
    int opt = 0;
    int opt_index = 0;
    int period_explicit = 0;
    int buffer_explicit = 0;
    size_t tag_width = 0;

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
                    fprintf(stderr, "mode invalido: %s (use hex, raw ou fft-raw)\n", optarg);
                    return -1;
                }
                break;
            case 'o':
                cfg->output_path = optarg;
                break;
            case 999:
                if (parse_telemetry_format(optarg, &cfg->telemetry_format) != 0) {
                    fprintf(stderr, "telemetry-format invalido: %s (use text ou jsonl)\n", optarg);
                    return -1;
                }
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
            case 1003:
                cfg->realign_initial_word_skip = 1U;
                cfg->realign_swap_channels = 1;
                cfg->show_tagged_fields = 1;
                break;
            case 1004:
                if (parse_nonneg_u32_arg(optarg, &cfg->realign_initial_word_skip) != 0) {
                    fprintf(stderr, "realign-initial-word-skip invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1005:
                cfg->realign_swap_channels = 1;
                break;
            case 1006:
                cfg->show_tagged_fields = 1;
                break;
            case 1007:
                if (parse_nonneg_u32_arg(optarg, &cfg->packet_index_shift) != 0) {
                    fprintf(stderr, "packet-index-shift invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1008:
                if (parse_u32_arg(optarg, &cfg->packet_index_bits) != 0) {
                    fprintf(stderr, "packet-index-bits invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1009:
                if (parse_u32_arg(optarg, &cfg->fft_packet_index_base) != 0) {
                    fprintf(stderr, "fft-packet-index-base invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1010:
                if (parse_nonneg_u32_arg(optarg, &cfg->tag_shift) != 0) {
                    fprintf(stderr, "tag-shift invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1011:
                if (parse_u32_arg(optarg, &cfg->tag_mask) != 0) {
                    fprintf(stderr, "tag-mask invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1012:
                if (parse_u32_arg(optarg, &cfg->payload_bits) != 0) {
                    fprintf(stderr, "payload-bits invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1013:
                if (parse_nonneg_u32_arg(optarg, &cfg->tag_idle) != 0) {
                    fprintf(stderr, "tag-idle invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1014:
                if (parse_nonneg_u32_arg(optarg, &cfg->tag_bfpexp) != 0) {
                    fprintf(stderr, "tag-bfpexp invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1015:
                if (parse_nonneg_u32_arg(optarg, &cfg->tag_fft) != 0) {
                    fprintf(stderr, "tag-fft invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1016:
                if (parse_u32_arg(optarg, &cfg->fft_frame_bins) != 0) {
                    fprintf(stderr, "fft-frame-bins invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1017:
                if (parse_u32_arg(optarg, &cfg->bfpexp_hold_pairs) != 0) {
                    fprintf(stderr, "bfpexp-hold-pairs invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1018:
                if (parse_nonneg_u32_arg(optarg, &cfg->loss_tolerance_pairs) != 0) {
                    fprintf(stderr, "loss-tolerance-pairs invalido: %s\n", optarg);
                    return -1;
                }
                break;
            case 1019:
                cfg->allow_fft_without_bfpexp = 1;
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
    if (cfg->packet_index_bits == 0U || cfg->packet_index_bits > 31U) {
        fprintf(stderr, "packet-index-bits deve ficar entre 1 e 31\n");
        return -1;
    }
    if (cfg->tag_mask == 0U) {
        fprintf(stderr, "tag-mask deve ser maior que zero\n");
        return -1;
    }
    if (cfg->payload_bits == 0U || cfg->payload_bits >= cfg->tag_shift) {
        fprintf(stderr, "payload-bits deve ficar entre 1 e tag-shift-1\n");
        return -1;
    }
    if (cfg->packet_index_shift > 31U || cfg->tag_shift > 31U) {
        fprintf(stderr, "packet-index-shift/tag-shift devem ficar entre 0 e 31\n");
        return -1;
    }
    tag_width = tag_mask_bit_width(cfg->tag_mask);
    if ((cfg->packet_index_shift + cfg->packet_index_bits) > 32U) {
        fprintf(stderr, "campo packet_index excede 32 bits\n");
        return -1;
    }
    if ((cfg->tag_shift + tag_width) > 32U) {
        fprintf(stderr, "campo tag excede 32 bits\n");
        return -1;
    }
    if (cfg->packet_index_shift < (cfg->tag_shift + tag_width)) {
        fprintf(stderr, "campo packet_index nao pode sobrepor o campo tag\n");
        return -1;
    }
    if (cfg->fft_frame_bins == 0U) {
        fprintf(stderr, "fft-frame-bins deve ser maior que zero\n");
        return -1;
    }
    if (cfg->fft_frame_bins > cfg->fft_packet_index_base) {
        fprintf(stderr, "fft-frame-bins deve caber no intervalo de packet_index FFT\n");
        return -1;
    }
    if (cfg->bfpexp_hold_pairs == 0U) {
        fprintf(stderr, "bfpexp-hold-pairs deve ser maior que zero\n");
        return -1;
    }
    if (cfg->bfpexp_hold_pairs > cfg->fft_packet_index_base) {
        fprintf(stderr, "bfpexp-hold-pairs deve caber no intervalo de packet_index BFPEXP\n");
        return -1;
    }
    if (realignment_enabled(cfg) && cfg->mode == OUTPUT_MODE_HEX) {
        cfg->show_tagged_fields = 1;
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

static uint32_t tagged_packet_index_mask(const capture_config_t *cfg) {
    return (uint32_t)((1ULL << cfg->packet_index_bits) - 1ULL);
}

static unsigned int decode_packet_index(const capture_config_t *cfg, uint32_t word) {
    return (unsigned int)((word >> cfg->packet_index_shift) & tagged_packet_index_mask(cfg));
}

static unsigned int decode_tag_value(const capture_config_t *cfg, uint32_t word) {
    return (unsigned int)((word >> cfg->tag_shift) & cfg->tag_mask);
}

static const char *decode_tag_name(const capture_config_t *cfg, unsigned int tag_value) {
    if (tag_value == cfg->tag_idle) {
        return "idle";
    }
    if (tag_value == cfg->tag_bfpexp) {
        return "bfpexp";
    }
    if (tag_value == cfg->tag_fft) {
        return "fft";
    }
    return "unknown";
}

static const char *classify_pair_kind(
    const capture_config_t *cfg,
    unsigned int left_tag,
    unsigned int right_tag,
    unsigned int left_packet_index,
    unsigned int right_packet_index
) {
    if (left_tag != right_tag) {
        return "tag_mismatch";
    }
    if (left_packet_index != right_packet_index) {
        return "packet_index_mismatch";
    }
    if (left_tag == cfg->tag_idle) {
        return left_packet_index == 0U ? "idle" : "packet_index_mismatch";
    }
    if (left_tag == cfg->tag_bfpexp) {
        return left_packet_index < cfg->fft_packet_index_base ? "bfpexp" : "packet_index_mismatch";
    }
    if (left_tag == cfg->tag_fft) {
        return left_packet_index >= cfg->fft_packet_index_base ? "fft" : "packet_index_mismatch";
    }
    return "unknown_tag";
}

static void format_bin_field(
    const capture_config_t *cfg,
    unsigned int tag_value,
    unsigned int packet_index,
    char *buffer,
    size_t buffer_size
) {
    if (tag_value == cfg->tag_fft && packet_index >= cfg->fft_packet_index_base) {
        (void)snprintf(buffer, buffer_size, "%u", packet_index - cfg->fft_packet_index_base);
        return;
    }
    (void)snprintf(buffer, buffer_size, "-");
}

enum {
    TAGGED_PAIR_KIND_LOSS = 0,
    TAGGED_PAIR_KIND_IDLE = 1,
    TAGGED_PAIR_KIND_BFPEXP = 2,
    TAGGED_PAIR_KIND_FFT = 3,
    FFT_RAW_HEADER_WORDS = 4,
    FFT_RAW_FLAG_EXPLICIT_BFPEXP = 1,
};

static int decode_payload_value(const capture_config_t *cfg, uint32_t word) {
    unsigned int payload_bits = cfg->payload_bits;
    uint32_t payload_mask = 0U;
    uint32_t payload = 0U;
    uint32_t sign_bit = 0U;

    if (payload_bits == 0U || payload_bits >= 32U) {
        return (int32_t)word;
    }

    payload_mask = (uint32_t)((1ULL << payload_bits) - 1ULL);
    payload = word & payload_mask;
    sign_bit = (uint32_t)(1U << (payload_bits - 1U));
    if ((payload & sign_bit) != 0U) {
        payload |= ~payload_mask;
    }
    return (int32_t)payload;
}

static int classify_pair_kind_code(
    const capture_config_t *cfg,
    unsigned int left_tag,
    unsigned int right_tag,
    unsigned int left_packet_index,
    unsigned int right_packet_index
) {
    const char *kind = classify_pair_kind(cfg, left_tag, right_tag, left_packet_index, right_packet_index);
    if (strcmp(kind, "idle") == 0) {
        return TAGGED_PAIR_KIND_IDLE;
    }
    if (strcmp(kind, "bfpexp") == 0) {
        return TAGGED_PAIR_KIND_BFPEXP;
    }
    if (strcmp(kind, "fft") == 0) {
        return TAGGED_PAIR_KIND_FFT;
    }
    return TAGGED_PAIR_KIND_LOSS;
}

static int ensure_tagged_fft_buffers(writer_context_t *writer) {
    tagged_fft_state_t *state = &writer->tagged_fft_state;
    size_t frame_word_count = 0;
    int needs_init = 0;

    if (writer->cfg->mode != OUTPUT_MODE_FFT_RAW) {
        return 0;
    }
    if (writer->cfg->fft_frame_bins == 0U) {
        return -1;
    }

    frame_word_count = (size_t)writer->cfg->fft_frame_bins * CHANNEL_COUNT;
    if (state->frame_pairs == NULL) {
        state->frame_pairs = calloc(frame_word_count, sizeof(int32_t));
        needs_init = 1;
    }
    if (state->received_bins == NULL) {
        state->received_bins = calloc((size_t)writer->cfg->fft_frame_bins, sizeof(uint8_t));
        needs_init = 1;
    }
    if (state->bfpexp_seen == NULL) {
        state->bfpexp_seen = calloc((size_t)writer->cfg->bfpexp_hold_pairs, sizeof(uint8_t));
        needs_init = 1;
    }
    if (state->frame_pairs == NULL || state->received_bins == NULL || state->bfpexp_seen == NULL) {
        return -1;
    }

    if (!needs_init) {
        return 0;
    }

    state->waiting_for_start = 1;
    state->current_bfpexp = 0;
    state->have_explicit_bfpexp = 0;
    state->bfpexp_gap_count = 0U;
    state->bfpexp_seen_count = 0U;
    state->highest_bin_index_seen = 0U;
    state->received_count = 0U;
    memset(state->frame_pairs, 0, frame_word_count * sizeof(int32_t));
    memset(state->received_bins, 0, (size_t)writer->cfg->fft_frame_bins * sizeof(uint8_t));
    memset(state->bfpexp_seen, 0, (size_t)writer->cfg->bfpexp_hold_pairs * sizeof(uint8_t));
    return 0;
}

static void reset_fft_frame_accumulator(writer_context_t *writer) {
    tagged_fft_state_t *state = &writer->tagged_fft_state;

    state->waiting_for_start = 1;
    state->have_explicit_bfpexp = 0;
    state->bfpexp_gap_count = 0U;
    state->bfpexp_seen_count = 0U;
    state->highest_bin_index_seen = 0U;
    state->received_count = 0U;
    if (state->frame_pairs != NULL) {
        memset(state->frame_pairs, 0, (size_t)writer->cfg->fft_frame_bins * CHANNEL_COUNT * sizeof(int32_t));
    }
    if (state->received_bins != NULL) {
        memset(state->received_bins, 0, (size_t)writer->cfg->fft_frame_bins * sizeof(uint8_t));
    }
    if (state->bfpexp_seen != NULL) {
        memset(state->bfpexp_seen, 0, (size_t)writer->cfg->bfpexp_hold_pairs * sizeof(uint8_t));
    }
}

static int emit_fft_raw_frame(writer_context_t *writer) {
    tagged_fft_state_t *state = &writer->tagged_fft_state;
    int32_t header[FFT_RAW_HEADER_WORDS];
    size_t payload_bytes = 0;

    if (state->frame_pairs == NULL || writer->out_fd < 0) {
        return 0;
    }
    if (state->received_count == 0U) {
        return 0;
    }

    header[0] = (int32_t)state->current_bfpexp;
    header[1] = state->have_explicit_bfpexp ? FFT_RAW_FLAG_EXPLICIT_BFPEXP : 0;
    header[2] = (int32_t)state->received_count;
    header[3] = 0;
    if (write_all(writer->out_fd, header, sizeof(header)) < 0) {
        return -1;
    }

    payload_bytes = (size_t)writer->cfg->fft_frame_bins * CHANNEL_COUNT * sizeof(int32_t);
    if (write_all(writer->out_fd, state->frame_pairs, payload_bytes) < 0) {
        return -1;
    }

    writer->stats->written_chunks += 1ULL;
    writer->stats->written_frames += (uint64_t)writer->cfg->fft_frame_bins;
    reset_fft_frame_accumulator(writer);
    return 0;
}

static int write_fft_raw_chunk(writer_context_t *writer, const int32_t *samples, snd_pcm_sframes_t frames) {
    const capture_config_t *cfg = writer->cfg;
    tagged_fft_state_t *state = &writer->tagged_fft_state;
    snd_pcm_sframes_t idx = 0;

    if (ensure_tagged_fft_buffers(writer) != 0) {
        return -1;
    }

    for (idx = 0; idx < frames; ++idx) {
        uint32_t left = (uint32_t)samples[(size_t)idx * CHANNEL_COUNT];
        uint32_t right = (uint32_t)samples[(size_t)idx * CHANNEL_COUNT + 1U];
        unsigned int packet_index = decode_packet_index(cfg, left);
        unsigned int left_tag = decode_tag_value(cfg, left);
        unsigned int right_tag = decode_tag_value(cfg, right);
        unsigned int right_packet_index = decode_packet_index(cfg, right);
        int payload_left = decode_payload_value(cfg, left);
        int payload_right = decode_payload_value(cfg, right);
        int reprocess_pair = 1;

        while (reprocess_pair) {
            int kind_code = classify_pair_kind_code(cfg, left_tag, right_tag, packet_index, right_packet_index);
            int bin_index = 0;
            int full_bfpexp_preamble = 0;

            reprocess_pair = 0;
            if (state->waiting_for_start) {
                if (kind_code == TAGGED_PAIR_KIND_IDLE) {
                    if (state->bfpexp_seen_count > 0U && state->bfpexp_gap_count < cfg->loss_tolerance_pairs) {
                        state->bfpexp_gap_count += 1U;
                    }
                    continue;
                }
                if (kind_code == TAGGED_PAIR_KIND_BFPEXP) {
                    state->current_bfpexp = payload_left;
                    state->have_explicit_bfpexp = 1;
                    if (packet_index == 0U) {
                        memset(state->bfpexp_seen, 0, (size_t)cfg->bfpexp_hold_pairs * sizeof(uint8_t));
                        state->bfpexp_seen[0] = 1U;
                        state->bfpexp_seen_count = 1U;
                        state->bfpexp_gap_count = 0U;
                    } else if (packet_index < cfg->bfpexp_hold_pairs && state->bfpexp_seen[packet_index] == 0U) {
                        state->bfpexp_seen[packet_index] = 1U;
                        state->bfpexp_seen_count += 1U;
                    }
                    continue;
                }
                if (kind_code == TAGGED_PAIR_KIND_LOSS) {
                    if (state->bfpexp_seen_count > 0U && state->bfpexp_gap_count < cfg->loss_tolerance_pairs) {
                        state->bfpexp_gap_count += 1U;
                    }
                    continue;
                }
                if (kind_code != TAGGED_PAIR_KIND_FFT) {
                    continue;
                }

                bin_index = (int)packet_index - (int)cfg->fft_packet_index_base;
                if (bin_index < 0 || (unsigned int)bin_index >= cfg->fft_frame_bins) {
                    continue;
                }

                full_bfpexp_preamble = (
                    state->bfpexp_seen_count > 0U &&
                    state->bfpexp_seen[0] != 0U &&
                    (state->bfpexp_seen_count + state->bfpexp_gap_count) >= cfg->bfpexp_hold_pairs
                );
                if (!full_bfpexp_preamble && !cfg->allow_fft_without_bfpexp) {
                    continue;
                }

                state->waiting_for_start = 0;
                state->received_count = 0U;
                state->highest_bin_index_seen = (unsigned int)bin_index;
                memset(state->frame_pairs, 0, (size_t)cfg->fft_frame_bins * CHANNEL_COUNT * sizeof(int32_t));
                memset(state->received_bins, 0, (size_t)cfg->fft_frame_bins * sizeof(uint8_t));
                state->frame_pairs[(size_t)bin_index * CHANNEL_COUNT] = (int32_t)payload_left;
                state->frame_pairs[(size_t)bin_index * CHANNEL_COUNT + 1U] = (int32_t)payload_right;
                state->received_bins[bin_index] = 1U;
                state->received_count = 1U;
                if (state->received_count >= cfg->fft_frame_bins && emit_fft_raw_frame(writer) != 0) {
                    return -1;
                }
                continue;
            }

            if (kind_code == TAGGED_PAIR_KIND_BFPEXP) {
                if (emit_fft_raw_frame(writer) != 0) {
                    return -1;
                }
                state->current_bfpexp = payload_left;
                state->have_explicit_bfpexp = 1;
                reprocess_pair = 1;
                continue;
            }
            if (kind_code == TAGGED_PAIR_KIND_FFT) {
                bin_index = (int)packet_index - (int)cfg->fft_packet_index_base;
                if (bin_index < 0 || (unsigned int)bin_index >= cfg->fft_frame_bins) {
                    continue;
                }
                if ((unsigned int)bin_index < state->highest_bin_index_seen) {
                    if (emit_fft_raw_frame(writer) != 0) {
                        return -1;
                    }
                    reprocess_pair = 1;
                    continue;
                }
                state->frame_pairs[(size_t)bin_index * CHANNEL_COUNT] = (int32_t)payload_left;
                state->frame_pairs[(size_t)bin_index * CHANNEL_COUNT + 1U] = (int32_t)payload_right;
                if (state->received_bins[bin_index] == 0U) {
                    state->received_bins[bin_index] = 1U;
                    state->received_count += 1U;
                }
                state->highest_bin_index_seen = (unsigned int)bin_index;
                if (state->received_count >= cfg->fft_frame_bins && emit_fft_raw_frame(writer) != 0) {
                    return -1;
                }
                continue;
            }
            if (kind_code == TAGGED_PAIR_KIND_IDLE || kind_code == TAGGED_PAIR_KIND_LOSS) {
                continue;
            }
        }
    }

    return 0;
}

static int ensure_transform_capacity(writer_context_t *writer, size_t required_words) {
    int32_t *new_buffer = NULL;
    size_t new_capacity = 0;

    if (required_words <= writer->transform_capacity_words) {
        return 0;
    }

    new_capacity = required_words;
    new_buffer = realloc(writer->transform_buffer, new_capacity * sizeof(int32_t));
    if (new_buffer == NULL) {
        return -1;
    }

    writer->transform_buffer = new_buffer;
    writer->transform_capacity_words = new_capacity;
    return 0;
}

static int realign_chunk(
    writer_context_t *writer,
    const int32_t *samples,
    snd_pcm_sframes_t frames,
    const int32_t **out_samples,
    snd_pcm_sframes_t *out_frames
) {
    const capture_config_t *cfg = writer->cfg;
    size_t input_words = (size_t)frames * CHANNEL_COUNT;
    size_t output_words = 0;
    size_t idx = 0;

    if (!realignment_enabled(cfg)) {
        *out_samples = samples;
        *out_frames = frames;
        return 0;
    }

    if (ensure_transform_capacity(writer, input_words + 1U) != 0) {
        return -1;
    }

    if (writer->realign_state.have_pending_word) {
        writer->transform_buffer[output_words++] = writer->realign_state.pending_word;
        writer->realign_state.have_pending_word = 0;
    }

    for (idx = 0; idx < input_words; ++idx) {
        if (writer->realign_state.initial_word_skip_remaining > 0U) {
            writer->realign_state.initial_word_skip_remaining -= 1U;
            continue;
        }
        writer->transform_buffer[output_words++] = samples[idx];
    }

    if ((output_words % CHANNEL_COUNT) != 0U) {
        writer->realign_state.pending_word = writer->transform_buffer[output_words - 1U];
        writer->realign_state.have_pending_word = 1;
        output_words -= 1U;
    }

    if (cfg->realign_swap_channels) {
        for (idx = 0; idx < output_words; idx += CHANNEL_COUNT) {
            int32_t tmp = writer->transform_buffer[idx];
            writer->transform_buffer[idx] = writer->transform_buffer[idx + 1U];
            writer->transform_buffer[idx + 1U] = tmp;
        }
    }

    *out_samples = writer->transform_buffer;
    *out_frames = (snd_pcm_sframes_t)(output_words / CHANNEL_COUNT);
    return 0;
}

static int write_hex_chunk(writer_context_t *writer, const int32_t *samples, snd_pcm_sframes_t frames) {
    FILE *output = writer->hex_output;
    const capture_config_t *cfg = writer->cfg;
    snd_pcm_sframes_t idx;

    for (idx = 0; idx < frames; ++idx) {
        uint32_t left = (uint32_t)samples[(size_t)idx * CHANNEL_COUNT];
        uint32_t right = (uint32_t)samples[(size_t)idx * CHANNEL_COUNT + 1U];
        if (!cfg->show_tagged_fields) {
            if (fprintf(output, "0x%08" PRIX32 " 0x%08" PRIX32 "\n", left, right) < 0) {
                return -1;
            }
            continue;
        }

        {
            unsigned int left_packet_index = decode_packet_index(cfg, left);
            unsigned int right_packet_index = decode_packet_index(cfg, right);
            unsigned int left_tag = decode_tag_value(cfg, left);
            unsigned int right_tag = decode_tag_value(cfg, right);
            int payload_left = decode_payload_value(cfg, left);
            int payload_right = decode_payload_value(cfg, right);
            char left_bin[32];
            char right_bin[32];
            const char *kind = classify_pair_kind(cfg, left_tag, right_tag, left_packet_index, right_packet_index);

            format_bin_field(cfg, left_tag, left_packet_index, left_bin, sizeof(left_bin));
            format_bin_field(cfg, right_tag, right_packet_index, right_bin, sizeof(right_bin));
            if (fprintf(output,
                        "0x%08" PRIX32 " 0x%08" PRIX32
                        " dec=%d/%d kind=%s pkt=%u/%u tag=%s/%s bin=%s/%s\n",
                        left,
                        right,
                        payload_left,
                        payload_right,
                        kind,
                        left_packet_index,
                        right_packet_index,
                        decode_tag_name(cfg, left_tag),
                        decode_tag_name(cfg, right_tag),
                        left_bin,
                        right_bin) < 0) {
                return -1;
            }
        }
    }

    return 0;
}

static void free_writer_buffers(writer_context_t *writer) {
    free(writer->transform_buffer);
    writer->transform_buffer = NULL;
    writer->transform_capacity_words = 0;
    free(writer->tagged_fft_state.frame_pairs);
    free(writer->tagged_fft_state.received_bins);
    free(writer->tagged_fft_state.bfpexp_seen);
    writer->tagged_fft_state.frame_pairs = NULL;
    writer->tagged_fft_state.received_bins = NULL;
    writer->tagged_fft_state.bfpexp_seen = NULL;
}

static void writer_emit_realign_summary(writer_context_t *writer) {
    if (!realignment_enabled(writer->cfg) || !writer->realign_state.have_pending_word) {
        return;
    }

    if (writer->cfg->telemetry_format == TELEMETRY_FORMAT_JSONL) {
        fprintf(stderr, "{\"type\":\"capture_warning\"");
        telemetry_write_u64_field(stderr, "timestamp_ns", realtime_ns());
        telemetry_write_string_field(stderr, "warning", "realign_trailing_word_dropped");
        telemetry_write_i64_field(stderr, "word_hex", (int64_t)((uint32_t)writer->realign_state.pending_word));
        fputs("}\n", stderr);
        fflush(stderr);
        return;
    }

    fprintf(stderr,
            "Aviso: realinhamento descartou a ultima palavra solta 0x%08" PRIX32 "\n",
            (uint32_t)writer->realign_state.pending_word);
}

static void writer_emit_fft_summary(writer_context_t *writer) {
    if (writer->mode != OUTPUT_MODE_FFT_RAW) {
        return;
    }
    if (emit_fft_raw_frame(writer) != 0 && writer->error_code == 0) {
        writer->error_code = errno != 0 ? errno : EIO;
    }
}

static void *writer_main(void *opaque) {
    writer_context_t *writer = (writer_context_t *)opaque;

    while (1) {
        size_t slot_index = 0;
        snd_pcm_sframes_t frames = 0;
        snd_pcm_sframes_t output_frames = 0;
        int32_t *samples = NULL;
        const int32_t *output_samples = NULL;

        if (queue_acquire_read_slot(writer->queue, &slot_index, &frames) != 0) {
            break;
        }

        samples = queue_slot_ptr(writer->queue, slot_index);
        if (realign_chunk(writer, samples, frames, &output_samples, &output_frames) != 0) {
            writer->error_code = ENOMEM;
            *writer->stop_flag = 1;
            queue_close(writer->queue);
            break;
        }

        if (output_frames > 0) {
            if (writer->mode == OUTPUT_MODE_RAW) {
                size_t chunk_bytes = (size_t)output_frames * CHANNEL_COUNT * sizeof(int32_t);
                if (write_all(writer->out_fd, output_samples, chunk_bytes) < 0) {
                    writer->error_code = errno != 0 ? errno : EIO;
                    *writer->stop_flag = 1;
                    queue_close(writer->queue);
                    break;
                }
                writer->stats->written_chunks += 1;
                writer->stats->written_frames += (uint64_t)output_frames;
            } else if (writer->mode == OUTPUT_MODE_FFT_RAW) {
                if (write_fft_raw_chunk(writer, output_samples, output_frames) != 0) {
                    writer->error_code = errno != 0 ? errno : EIO;
                    *writer->stop_flag = 1;
                    queue_close(writer->queue);
                    break;
                }
            } else {
                if (write_hex_chunk(writer, output_samples, output_frames) != 0) {
                    writer->error_code = errno != 0 ? errno : EIO;
                    *writer->stop_flag = 1;
                    queue_close(writer->queue);
                    break;
                }
                if (writer->flush_every_chunks > 0 &&
                    (((writer->stats->written_chunks + 1ULL) % writer->flush_every_chunks) == 0ULL)) {
                    fflush(writer->hex_output);
                }
                writer->stats->written_chunks += 1;
                writer->stats->written_frames += (uint64_t)output_frames;
            }
        }

        queue_release_read_slot(writer->queue);
    }

    writer_emit_fft_summary(writer);
    writer_emit_realign_summary(writer);
    if (writer->mode == OUTPUT_MODE_HEX && writer->hex_output != NULL) {
        fflush(writer->hex_output);
    }
    return NULL;
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

    if (cfg->mode == OUTPUT_MODE_RAW || cfg->mode == OUTPUT_MODE_FFT_RAW) {
        if (cfg->output_path == NULL || strcmp(cfg->output_path, "-") == 0) {
            if (isatty(STDOUT_FILENO)) {
                fprintf(stderr, "Modo raw/fft-raw nao pode escrever no terminal. Use --output arquivo.raw ou pipe.\n");
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
    free_writer_buffers(writer);
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

static int recover_capture_error(
    snd_pcm_t *pcm,
    int err,
    const capture_config_t *cfg,
    capture_queue_t *queue,
    capture_stats_t *stats,
    const char *context
) {
    int recovered;

    if (err == -EPIPE) {
        stats->xruns += 1;
    }

    recovered = snd_pcm_recover(pcm, err, 1);
    if (recovered < 0) {
        emit_capture_recovery(cfg, queue, stats, context, err, recovered);
        fprintf(stderr, "%s: %s\n", context, snd_strerror(recovered));
        return recovered;
    }

    stats->recoveries += 1;
    emit_capture_recovery(cfg, queue, stats, context, err, recovered);
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

    if (cfg->telemetry_format == TELEMETRY_FORMAT_JSONL) {
        fprintf(stderr, "{\"type\":\"%s\"", final_report ? "capture_final" : "capture_stats");
        telemetry_write_u64_field(stderr, "timestamp_ns", realtime_ns());
        telemetry_write_string_field(stderr, "mode", output_mode_name(cfg->mode));
        telemetry_write_u64_field(stderr, "rate_hz", (uint64_t)cfg->rate_hz);
        telemetry_write_u64_field(stderr, "captured_chunks", stats->captured_chunks);
        telemetry_write_u64_field(stderr, "captured_frames", stats->captured_frames);
        telemetry_write_u64_field(stderr, "written_chunks", stats->written_chunks);
        telemetry_write_u64_field(stderr, "written_frames", stats->written_frames);
        telemetry_write_u64_field(stderr, "xruns", stats->xruns);
        telemetry_write_u64_field(stderr, "recoveries", stats->recoveries);
        telemetry_write_u64_field(stderr, "partial_reads", stats->partial_reads);
        telemetry_write_u64_field(stderr, "queue_used_slots", (uint64_t)used_slots);
        telemetry_write_u64_field(stderr, "queue_slot_count", (uint64_t)queue->slot_count);
        telemetry_write_u64_field(stderr, "queue_high_water_slots", (uint64_t)high_water_slots);
        telemetry_write_u64_field(stderr, "queue_full_waits", full_waits);
        fputs("}\n", stderr);
        fflush(stderr);
        return;
    }

    fprintf(stderr,
            "[%s] mode=%s rate=%u captured_chunks=%" PRIu64 " captured_frames=%" PRIu64
            " written_chunks=%" PRIu64 " written_frames=%" PRIu64
            " xruns=%" PRIu64 " recoveries=%" PRIu64 " partial_reads=%" PRIu64
            " queue=%zu/%zu queue_high=%zu full_waits=%" PRIu64 "\n",
            label,
            output_mode_name(cfg->mode),
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
        .telemetry_format = TELEMETRY_FORMAT_TEXT,
        .realign_initial_word_skip = 0U,
        .realign_swap_channels = 0,
        .show_tagged_fields = 0,
        .packet_index_shift = DEFAULT_PACKET_INDEX_SHIFT,
        .packet_index_bits = DEFAULT_PACKET_INDEX_BITS,
        .fft_packet_index_base = DEFAULT_FFT_PACKET_INDEX_BASE,
        .tag_shift = DEFAULT_TAG_SHIFT,
        .tag_mask = DEFAULT_TAG_MASK,
        .payload_bits = DEFAULT_PAYLOAD_BITS,
        .tag_idle = DEFAULT_TAG_IDLE,
        .tag_bfpexp = DEFAULT_TAG_BFPEXP,
        .tag_fft = DEFAULT_TAG_FFT,
        .fft_frame_bins = DEFAULT_READ_FRAMES,
        .bfpexp_hold_pairs = 128U,
        .loss_tolerance_pairs = 3U,
        .allow_fft_without_bfpexp = 0,
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
    unsigned int requested_rate_hz = 0;

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
    requested_rate_hz = cfg.rate_hz;

    if (queue_init(&queue, cfg.queue_chunks, cfg.read_frames) != 0) {
        return 1;
    }

    writer.queue = &queue;
    writer.stop_flag = &g_stop;
    writer.mode = cfg.mode;
    writer.flush_every_chunks = cfg.flush_every_chunks;
    writer.stats = &stats;
    writer.cfg = &cfg;
    writer.realign_state.initial_word_skip_remaining = cfg.realign_initial_word_skip;

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
    if (cfg.rate_hz != requested_rate_hz) {
        emit_capture_warning_rate_adjusted(&cfg, requested_rate_hz);
    }
    if ((err = set_sw_params(pcm, cfg.period_frames, cfg.buffer_frames)) < 0) {
        goto cleanup;
    }
    if ((err = snd_pcm_prepare(pcm)) < 0) {
        fprintf(stderr, "snd_pcm_prepare: %s\n", snd_strerror(err));
        goto cleanup;
    }

    emit_capture_session_start(&cfg, &queue, requested_rate_hz);

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
            if (recover_capture_error(pcm, (int)got, &cfg, &queue, &stats, "snd_pcm_readi") < 0) {
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
        emit_capture_chunk(&cfg, &queue, &stats, got, (snd_pcm_uframes_t)got < cfg.read_frames);
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
