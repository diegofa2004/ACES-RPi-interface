import numpy as np
import time
import os

def compararEvento(buffer2, buffer4, lock, get_lastEventTime):
    COOLDOWN = 15
    PASSO_JANELA = 2
    LIMIAR_SCORE = 0.5
    FRAMES_MIN_DETECCAO = 3

    detectando = False
    contador = 0

    evento_mfcc_cache = None
    evento_fft_cache = None
    last_mtime_evento = 0
    last_mtime_fft = 0

    def cosine_sim(a, b):
        num = np.sum(a * b)
        den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(num / den)

    def zscore_cols(x):
        media = np.mean(x, axis=0, keepdims=True)
        desvio = np.std(x, axis=0, keepdims=True) + 1e-9
        return (x - media) / desvio

    def zscore_rows(x):
        media = np.mean(x, axis=1, keepdims=True)
        desvio = np.std(x, axis=1, keepdims=True) + 1e-9
        return (x - media) / desvio

    def prepara_mfcc(m):
        m = np.asarray(m, dtype=np.float32)
        m = m[:, :8]
        return zscore_cols(m)

    def prepara_fft(f):
        f = np.asarray(f, dtype=np.float32)

        # comprime a escala da magnitude
        f = np.log1p(f)

        # mantém só a parte mais útil do espectro
        # para FFT_SIZE=2048, rfft gera 1025 bins
        # usar os primeiros 256 já costuma funcionar bem
        f = f[:, :256]

        # normaliza cada frame separadamente
        f = zscore_rows(f)
        return f

    def sim_mfcc(evento_mfcc, janela_mfcc):
        sims = []
        for k in range(evento_mfcc.shape[1]):
            sims.append(cosine_sim(evento_mfcc[:, k], janela_mfcc[:, k]))
        return float(np.mean(sims))

    def sim_fft(evento_fft, janela_fft):
        sims = []
        for t in range(evento_fft.shape[0]):
            sims.append(cosine_sim(evento_fft[t], janela_fft[t]))
        return float(np.mean(sims))

    while True:
        time.sleep(0.5)

        if time.time() - get_lastEventTime() < COOLDOWN:
            continue

        if not os.path.exists("evento.npy") or not os.path.exists("fft.npy"):
            continue

        mtime_evento = os.path.getmtime("evento.npy")
        mtime_fft = os.path.getmtime("fft.npy")

        if (
            evento_mfcc_cache is None
            or evento_fft_cache is None
            or mtime_evento != last_mtime_evento
            or mtime_fft != last_mtime_fft
        ):
            print("evento atualizado!")

            evento_mfcc_raw = np.load("evento.npy")
            evento_fft_raw = np.load("fft.npy")

            evento_mfcc_cache = prepara_mfcc(evento_mfcc_raw)
            evento_fft_cache = prepara_fft(evento_fft_raw)

            last_mtime_evento = mtime_evento
            last_mtime_fft = mtime_fft

            print("shape evento mfcc:", evento_mfcc_cache.shape)
            print("shape evento fft:", evento_fft_cache.shape)

        evento_mfcc = evento_mfcc_cache
        evento_fft = evento_fft_cache
        tam_evento = len(evento_mfcc)

        with lock:
            snapshot_mfcc = list(buffer2)
            snapshot_fft = list(buffer4)

        if len(snapshot_mfcc) < tam_evento or len(snapshot_fft) < tam_evento:
            continue

        atual_mfcc_full = np.array(snapshot_mfcc, dtype=np.float32)
        atual_fft_full = np.array(snapshot_fft, dtype=np.float32)

        melhor_score = -1.0
        melhor_sim_mfcc = -1.0
        melhor_sim_fft = -1.0

        for i in range(0, len(atual_mfcc_full) - tam_evento + 1, PASSO_JANELA):
            janela_mfcc_raw = atual_mfcc_full[i:i + tam_evento]
            janela_fft_raw = atual_fft_full[i:i + tam_evento]

            janela_mfcc = prepara_mfcc(janela_mfcc_raw)
            janela_fft = prepara_fft(janela_fft_raw)

            s_mfcc = sim_mfcc(evento_mfcc, janela_mfcc)
            s_fft = sim_fft(evento_fft, janela_fft)

            # dá um pouco mais de peso para FFT
            score = 0.40 * s_mfcc + 0.60 * s_fft

            if score > melhor_score:
                melhor_score = score
                melhor_sim_mfcc = s_mfcc
                melhor_sim_fft = s_fft

        print(
            f"score={melhor_score:.3f} "
            f"mfcc={melhor_sim_mfcc:.3f} "
            f"fft={melhor_sim_fft:.3f}"
        )

        if melhor_score > LIMIAR_SCORE:
            contador += 1
        else:
            contador = 0

        if contador >= FRAMES_MIN_DETECCAO:
            if not detectando:
                print("OPA! SOM SEMELHANTE!!!")
                detectando = True
        else:
            detectando = False