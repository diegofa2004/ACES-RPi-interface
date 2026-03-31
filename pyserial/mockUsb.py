import sounddevice as sd
import serial
import numpy as np

PORT = "/tmp/ttyE"
BAUDRATE = 1000000
SAMPLE_RATE = 44100

ser = serial.Serial(PORT, BAUDRATE, timeout=1)

def callback(indata, frames, time, status):

    samples = (indata * 32767).astype(np.int16)
    ser.write(samples.tobytes())

with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize = 1024,
        callback=callback):

    print("Enviando áudio para serial...")
    input("Pressione ENTER para parar\n")

ser.close()
