import os
import time
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


EPSILON = 1e-9


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - np.mean(x, dtype=np.float32)) / (np.std(x, dtype=np.float32) + EPSILON)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    media = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    desvio = np.std(x, axis=1, keepdims=True, dtype=np.float32) + EPSILON
    return (x - media) / desvio


def _cosine_batch(batch: np.ndarray, ref: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    numerador = batch @ ref
    denominador = (np.linalg.norm(batch, axis=1) * np.linalg.norm(ref)) + EPSILON
    return numerador / denominador


def _prepara_mfcc(m: np.ndarray) -> np.ndarray:
    return np.asarray(m, dtype=np.float32)[:, :8]


def _prepara_fft(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float32)
    f = np.log1p(f[:, :256])
    noise_floor = np.percentile(f, 20, axis=1, keepdims=True)
    f = f - noise_floor
    return np.maximum(f, 0.0)


def _energia_frames_fft_raw(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float32)
    return np.sum(np.log1p(f[:, :256]), axis=1, dtype=np.float32)


def _assinatura_fft(f_sel: np.ndarray) -> np.ndarray:
    return _zscore_1d(np.mean(f_sel, axis=0, dtype=np.float32))


def _assinatura_mfcc(m_sel: np.ndarray) -> np.ndarray:
    return _zscore_1d(np.mean(m_sel, axis=0, dtype=np.float32))


def _assinatura_env(f_sel_raw: np.ndarray) -> np.ndarray:
    return _zscore_1d(_energia_frames_fft_raw(f_sel_raw))


def _window_mean_2d(x: np.ndarray, window_size: int, step: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    prefix = np.zeros((x.shape[0] + 1, x.shape[1]), dtype=np.float32)
    prefix[1:] = np.cumsum(x, axis=0, dtype=np.float32)
    window_sum = prefix[window_size:] - prefix[:-window_size]
    return window_sum[::step] / np.float32(window_size)


def _window_mean_1d(x: np.ndarray, window_size: int, step: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    prefix = np.zeros(x.shape[0] + 1, dtype=np.float32)
    prefix[1:] = np.cumsum(x, dtype=np.float32)
    window_sum = prefix[window_size:] - prefix[:-window_size]
    return window_sum[::step] / np.float32(window_size)


def _melhor_bloco_continuo(energia: np.ndarray, min_frac: float = 0.12, max_frac: float = 0.30) -> tuple[int, int]:
    n = len(energia)
    if n < 8:
        return 0, n

    min_len = max(8, int(n * min_frac))
    max_len = max(min_len, int(n * max_frac))

    prefix = np.zeros(n + 1, dtype=np.float32)
    prefix[1:] = np.cumsum(energia, dtype=np.float32)

    melhor_i = 0
    melhor_j = min_len
    melhor_media = -1.0

    for window_len in range(min_len, max_len + 1):
        window_sum = prefix[window_len:] - prefix[:-window_len]
        if window_sum.size == 0:
            continue
        idx = int(np.argmax(window_sum))
        media = float(window_sum[idx] / np.float32(window_len))
        if media > melhor_media:
            melhor_media = media
            melhor_i = idx
            melhor_j = idx + window_len

    return melhor_i, melhor_j


def compararEvento(buffer2, buffer4, lock, get_lastEventTime):
    COOLDOWN = 15
    PASSO_JANELA = 1
    FRAMES_MIN_DETECCAO = 2
    MIN_ENERGIA_JANELA = 0.20
    POLL_INTERVAL_SECONDS = 0.20

    detectando = False
    contador = 0

    hist_scores = deque(maxlen=30)

    bloco_mfcc_ref = None
    bloco_fft_ref = None
    bloco_env_ref = None
    bloco_tamanho_cache = 0
    energia_bloco_ref_cache = 0.0
    last_mtime_evento = 0
    last_mtime_fft = 0

    while True:
        time.sleep(POLL_INTERVAL_SECONDS)

        if time.time() - get_lastEventTime() < COOLDOWN:
            continue

        if not os.path.exists("evento.npy") or not os.path.exists("fft.npy"):
            continue

        mtime_evento = os.path.getmtime("evento.npy")
        mtime_fft = os.path.getmtime("fft.npy")

        if (
            bloco_mfcc_ref is None
            or bloco_fft_ref is None
            or bloco_env_ref is None
            or mtime_evento != last_mtime_evento
            or mtime_fft != last_mtime_fft
        ):
            print("evento atualizado!")

            evento_mfcc_raw = np.load("evento.npy")
            evento_fft_raw = np.load("fft.npy").astype(np.float32)

            energia_total = _energia_frames_fft_raw(evento_fft_raw)
            i0, i1 = _melhor_bloco_continuo(energia_total)

            bloco_mfcc = _prepara_mfcc(evento_mfcc_raw)[i0:i1]
            bloco_fft = _prepara_fft(evento_fft_raw)[i0:i1]
            bloco_fft_raw = evento_fft_raw[i0:i1]

            bloco_mfcc_ref = _assinatura_mfcc(bloco_mfcc)
            bloco_fft_ref = _assinatura_fft(bloco_fft)
            bloco_env_ref = _assinatura_env(bloco_fft_raw)

            bloco_tamanho_cache = i1 - i0
            energia_bloco_ref_cache = float(np.mean(_energia_frames_fft_raw(bloco_fft_raw), dtype=np.float32) + EPSILON)

            last_mtime_evento = mtime_evento
            last_mtime_fft = mtime_fft
            hist_scores.clear()
            contador = 0
            detectando = False

            print(f"bloco usado: {i0}:{i1} tamanho={bloco_tamanho_cache}")

        bloco_tamanho = bloco_tamanho_cache
        energia_bloco_ref = energia_bloco_ref_cache

        with lock:
            snapshot_mfcc = list(buffer2)
            snapshot_fft = list(buffer4)

        if len(snapshot_mfcc) < bloco_tamanho or len(snapshot_fft) < bloco_tamanho:
            continue

        loop_start = time.perf_counter()

        atual_mfcc_full = np.asarray(snapshot_mfcc, dtype=np.float32)
        atual_fft_full = np.asarray(snapshot_fft, dtype=np.float32)

        energia_frames = _energia_frames_fft_raw(atual_fft_full)
        energia_janelas = _window_mean_1d(energia_frames, bloco_tamanho, PASSO_JANELA)
        validas = energia_janelas >= (MIN_ENERGIA_JANELA * energia_bloco_ref)

        mfcc_medias = _window_mean_2d(_prepara_mfcc(atual_mfcc_full), bloco_tamanho, PASSO_JANELA)
        fft_medias = _window_mean_2d(_prepara_fft(atual_fft_full), bloco_tamanho, PASSO_JANELA)
        env_janelas = sliding_window_view(energia_frames, bloco_tamanho)[::PASSO_JANELA]

        ref_mfcc = _zscore_rows(mfcc_medias)
        ref_fft = _zscore_rows(fft_medias)
        ref_env = _zscore_rows(np.asarray(env_janelas, dtype=np.float32))

        s_mfcc = _cosine_batch(ref_mfcc, bloco_mfcc_ref)
        s_fft = _cosine_batch(ref_fft, bloco_fft_ref)
        s_env = _cosine_batch(ref_env, bloco_env_ref)

        score = (0.10 * s_mfcc) + (0.65 * s_fft) + (0.25 * s_env)
        score = np.where(validas, score, -np.inf)

        if np.all(~validas):
            melhor_score = 0.0
            melhor_mfcc = 0.0
            melhor_fft = 0.0
            melhor_env = 0.0
        else:
            melhor_idx = int(np.argmax(score))
            melhor_score = float(score[melhor_idx])
            melhor_mfcc = float(s_mfcc[melhor_idx])
            melhor_fft = float(s_fft[melhor_idx])
            melhor_env = float(s_env[melhor_idx])

        hist_scores.append(melhor_score)

        if len(hist_scores) >= 10:
            arr = np.asarray(hist_scores, dtype=np.float32)
            baseline = float(np.median(arr))
            spread = float(np.std(arr, dtype=np.float32) + EPSILON)
        else:
            baseline = 0.0
            spread = 1.0

        score_rel = (melhor_score - baseline) / spread
        elapsed_ms = (time.perf_counter() - loop_start) * 1000.0

        print(
            f"score={melhor_score:.3f} "
            f"rel={score_rel:.3f} "
            f"base={baseline:.3f} "
            f"mfcc={melhor_mfcc:.3f} "
            f"fft={melhor_fft:.3f} "
            f"env={melhor_env:.3f} "
            f"janelas={len(score)} "
            f"proc_ms={elapsed_ms:.1f}"
        )

        if score_rel > 1.8:
            contador += 1
        else:
            contador = 0

        if contador >= FRAMES_MIN_DETECCAO:
            if not detectando:
                print("OPA! SOM SEMELHANTE!!!")
                detectando = True
        else:
            detectando = False
