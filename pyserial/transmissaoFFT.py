import numpy as np
import time
import os
from collections import deque

def compararEvento(buffer2, buffer4, lock, get_lastEventTime):
    COOLDOWN = 15
    PASSO_JANELA = 1
    FRAMES_MIN_DETECCAO = 2
    MIN_ENERGIA_JANELA = 0.20

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

    def cosine_sim(a, b):
        num = np.sum(a * b)
        den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(num / den)

    def zscore_1d(x):
        x = np.asarray(x, dtype=np.float32)
        return (x - np.mean(x)) / (np.std(x) + 1e-9)

    def prepara_mfcc(m):
        return np.asarray(m, dtype=np.float32)[:, :8]

    def prepara_fft(f):
        f = np.asarray(f, dtype=np.float32)
        f = np.log1p(f)
        f = f[:, :256]

        noise_floor = np.percentile(f, 20, axis=1, keepdims=True)
        f = f - noise_floor
        f = np.maximum(f, 0.0)
        return f

    def energia_frames_fft_raw(f):
        f = np.asarray(f, dtype=np.float32)
        f = np.log1p(f[:, :256])
        return np.sum(f, axis=1)

    def assinatura_fft(f_sel):
        # média espectral do trecho
        v = np.mean(f_sel, axis=0)
        return zscore_1d(v)

    def assinatura_mfcc(m_sel):
        v = np.mean(m_sel, axis=0)
        return zscore_1d(v)

    def assinatura_env(f_sel_raw):
        env = energia_frames_fft_raw(f_sel_raw)
        return zscore_1d(env)

    def melhor_bloco_continuo(energia, min_frac=0.12, max_frac=0.30):
        n = len(energia)
        if n < 8:
            return 0, n

        min_len = max(8, int(n * min_frac))
        max_len = max(min_len, int(n * max_frac))

        prefix = np.zeros(n + 1, dtype=np.float32)
        prefix[1:] = np.cumsum(energia)

        melhor_i = 0
        melhor_j = min_len
        melhor_media = -1.0

        for L in range(min_len, max_len + 1):
            for i in range(0, n - L + 1):
                j = i + L
                soma = prefix[j] - prefix[i]
                media = soma / L
                if media > melhor_media:
                    melhor_media = media
                    melhor_i = i
                    melhor_j = j

        return melhor_i, melhor_j

    while True:
        time.sleep(0.20)

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

            energia_total = energia_frames_fft_raw(evento_fft_raw)
            i0, i1 = melhor_bloco_continuo(energia_total)

            bloco_mfcc = prepara_mfcc(evento_mfcc_raw)[i0:i1]
            bloco_fft = prepara_fft(evento_fft_raw)[i0:i1]
            bloco_fft_raw = evento_fft_raw[i0:i1]

            bloco_mfcc_ref = assinatura_mfcc(bloco_mfcc)
            bloco_fft_ref = assinatura_fft(bloco_fft)
            bloco_env_ref = assinatura_env(bloco_fft_raw)

            bloco_tamanho_cache = i1 - i0
            energia_bloco_ref_cache = float(np.mean(energia_frames_fft_raw(bloco_fft_raw)) + 1e-9)

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

        atual_mfcc_full = np.array(snapshot_mfcc, dtype=np.float32)
        atual_fft_full = np.array(snapshot_fft, dtype=np.float32)

        melhor_score = -1.0
        melhor_mfcc = -1.0
        melhor_fft = -1.0
        melhor_env = -1.0

        for i in range(0, len(atual_mfcc_full) - bloco_tamanho + 1, PASSO_JANELA):
            janela_mfcc_raw = atual_mfcc_full[i:i + bloco_tamanho]
            janela_fft_raw = atual_fft_full[i:i + bloco_tamanho]

            energia_janela = float(np.mean(energia_frames_fft_raw(janela_fft_raw)))
            if energia_janela < MIN_ENERGIA_JANELA * energia_bloco_ref:
                continue

            janela_mfcc = prepara_mfcc(janela_mfcc_raw)
            janela_fft = prepara_fft(janela_fft_raw)

            ref_mfcc = assinatura_mfcc(janela_mfcc)
            ref_fft = assinatura_fft(janela_fft)
            ref_env = assinatura_env(janela_fft_raw)

            s_mfcc = cosine_sim(bloco_mfcc_ref, ref_mfcc)
            s_fft = cosine_sim(bloco_fft_ref, ref_fft)
            s_env = cosine_sim(bloco_env_ref, ref_env)

            score = 0.10 * s_mfcc + 0.65 * s_fft + 0.25 * s_env

            if score > melhor_score:
                melhor_score = score
                melhor_mfcc = s_mfcc
                melhor_fft = s_fft
                melhor_env = s_env

        if melhor_score < 0:
            melhor_score = 0.0
            melhor_mfcc = 0.0
            melhor_fft = 0.0
            melhor_env = 0.0

        hist_scores.append(melhor_score)

        if len(hist_scores) >= 10:
            arr = np.array(hist_scores, dtype=np.float32)
            baseline = float(np.median(arr))
            spread = float(np.std(arr) + 1e-9)
        else:
            baseline = 0.0
            spread = 1.0

        score_rel = (melhor_score - baseline) / spread

        print(
            f"score={melhor_score:.3f} "
            f"rel={score_rel:.3f} "
            f"base={baseline:.3f} "
            f"mfcc={melhor_mfcc:.3f} "
            f"fft={melhor_fft:.3f} "
            f"env={melhor_env:.3f}"
        )

        # detecção relativa, não absoluta
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