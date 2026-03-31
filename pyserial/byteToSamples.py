import numpy as np
def byteToSamples(data):
    return np.frombuffer(data, dtype=np.int16)