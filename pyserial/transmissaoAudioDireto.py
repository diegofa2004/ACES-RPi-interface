import serial
import threading
from collections import deque
import time
import librosa
import os
import numpy as np
from byteToSamples import byteToSamples 
from calculaFFT import calculaFFT
from calculaMFCC import calculaMFCC
from compararEvento import compararEvento
record_start = None

# serial
PORT = '/tmp/ttyS'
BAUDRATE = 1000000  

# audio
SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2
OUTPUT_FILE = "evento.npy"

lock = threading.Lock()
lastEventTime = 0
CHUNK = 4096
FFT_SIZE = 2048
# 5 segundos de buffer
BUFFER_SECONDS = 5
BUFFER_SIZE = 110

#5 segundos de buffer (FFT)
BUFFER_SECONDS3 = 5
BUFFER_SIZE3 = 110
 

#15 segundos de buffer(FFT) 
BUFFER_SECONDS4 = 15
BUFFER_SIZE4= 330


# buffer circular
buffer = deque(maxlen=BUFFER_SIZE)
buffer3 = deque(maxlen=BUFFER_SIZE3)

#15 segundos de buffer
BUFFER_SECONDS2 = 15
BUFFER_SIZE2 = 330  


# buffer 15 segundos
buffer2 = deque(maxlen=BUFFER_SIZE2)
buffer4 = deque(maxlen=BUFFER_SIZE4)

recording = False
ser = serial.Serial(PORT, BAUDRATE, timeout=1)
mel_filter = librosa.filters.mel(
    sr=SAMPLE_RATE,
    n_fft=FFT_SIZE,
    n_mels=32
)
print("ENTER → gravar / pausar")
print("Ctrl+C → sair")

def toggle_recording():
    global recording, record_start, future_buffer, lastEventTime, future_buffer2

    while True:
        input()

        if not recording:
            print("🔴 Gravando (incluindo últimos 5 s)...")

            # salva o buffer antes de começar
            #codigo salva buffer
            future_buffer = []
            future_buffer2 = []
            recording = True
            record_start = time.time()

threading.Thread(target=toggle_recording, daemon=True).start()
threading.Thread(target=compararEvento,args=(buffer2,buffer4,lock,lambda:lastEventTime), daemon=True).start()


try:
    while True:

        data = ser.read(CHUNK)
        if not data:
            continue
        samples = byteToSamples(data)
        if len(samples) < FFT_SIZE:
            continue
        fftBins = calculaFFT(samples)
        mfcc = calculaMFCC(fftBins, mel_filter)
        # sempre atualiza buffer circular
        mfcc = mfcc[:8]
        with lock:
            buffer.append(mfcc)
            buffer2.append(mfcc)    
            buffer3.append(fftBins)
            buffer4.append(fftBins)

        if recording:
            future_buffer.append(mfcc)
            future_buffer2.append(fftBins)
            if time.time() - record_start >= 5:
                evento = list(buffer) + future_buffer
                evento = np.array(evento)
                fft = list(buffer3) + future_buffer2
                fft = np.array(fft)
                np.save("evento_tmp.npy", evento)
                os.replace("evento_tmp.npy", "evento.npy")
                np.save("fft_tmp.npy", fft)
                os.replace("fft_tmp.npy", "fft.npy")
                lastEventTime = time.time()
                recording = False
                print("evento de 10s salvo")
    
except KeyboardInterrupt:
    print("\nEncerrando")

ser.close()


print("Arquivo salvo:", OUTPUT_FILE)
