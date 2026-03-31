import numpy as np
def calculaFFT(samples):
    window = np.hanning(len(samples))
    samples = samples * window
    fft = np.fft.rfft(samples)
    fftBins= np.abs(fft)
    return fftBins