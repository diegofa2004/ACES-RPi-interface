import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os

fig, ax = plt.subplots()
plt.ion()
plt.show(block=False)

fft_cache = None
last_mtime = 0
img = None

SAMPLE_RATE = 44100
FFT_SIZE = 2048
MAX_FREQ = 20000
PASSO_HZ = 1000

freq_por_bin = SAMPLE_RATE / FFT_SIZE
max_bin = int(MAX_FREQ / freq_por_bin)

while True:
    if not os.path.exists("fft.npy"):
        plt.pause(0.1)
        continue

    mtime = os.path.getmtime("fft.npy")

    if fft_cache is None or mtime != last_mtime:
        print("🔄 FFT atualizada!")

        fft_cache = np.load("fft.npy")
        print(fft_cache.shape)
        print("min", np.min(fft_cache))
        print("max", np.max(fft_cache))
        print("std total", np.std(fft_cache))
        print("std frame 0", np.std(np.abs(fft_cache[0])))

        last_mtime = mtime

        fft_mag = np.abs(fft_cache)
        fft_mag = 20 * np.log10(fft_mag + 1e-9)
        fft_mag = fft_mag[:, :max_bin]

        vmax = np.max(fft_mag)
        vmin = vmax - 50

        print("min log", np.min(fft_mag))
        print("max log", np.max(fft_mag))

        ax.clear()

        img = ax.imshow(
            fft_mag.T,
            aspect='auto',
            origin='lower',
            cmap='inferno',
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title("FFT ao longo do tempo")
        ax.set_xlabel("Tempo (frames)")
        ax.set_ylabel("Frequência (Hz)")

        freqs_hz = np.arange(0, MAX_FREQ + 1, PASSO_HZ)
        yticks = freqs_hz / freq_por_bin
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(f)}" for f in freqs_hz])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    else:
        plt.pause(0.1)