import numpy as np
from scipy.fftpack import dct

def calculaMFCC(fftBins, mel_filter):

    mel = mel_filter @ fftBins

    mel = np.log(mel + 1e-9)

    mfcc = dct(mel, type=2, norm='ortho')

    return mfcc[:13]   # primeiros 13 coeficientes
