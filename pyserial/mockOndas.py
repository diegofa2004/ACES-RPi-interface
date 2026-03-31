import numpy as np
import serial
import time
PORT = '/tmp/ttyE'
BAUD_RATE = 1000000
SAMPLE_RATE = 44100
FREQ = 2000
CHUNK = 1024
ser = serial.Serial(PORT,BAUD_RATE)
t = 0
print("enviando senoide...")
try: 
    while True:
        t_vals = (np.arange(CHUNK) + t) / SAMPLE_RATE
        samples = 0.5 * np.sin(2*np.pi*FREQ* t_vals)
        samples_int16 = (samples * 32767).astype(np.int16)
        ser.write(samples_int16.tobytes())
        t += CHUNK
        time.sleep(CHUNK / SAMPLE_RATE)
except KeyboardInterrupt:
    print("\nencerrando")
ser.close()