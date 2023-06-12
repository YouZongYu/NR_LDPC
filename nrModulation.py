import numpy as np

def nrSymbolModulate(enc, modulation):
    if modulation == 'QPSK':
        chIn = ((enc[0::2]*(-2) + 1) + 1j * (enc[1::2]*(-2) + 1))/np.sqrt(2)
    return chIn

def nrSymbolDemodulate(chOut, noiseVar, modulation):
    if modulation == 'QPSK':
        LLR = np.zeros((chOut.size*2,1))
        LLR[0::2] = np.real(chOut)*(2*np.sqrt(2)/noiseVar)
        LLR[1::2] = np.imag(chOut)*(2*np.sqrt(2)/noiseVar)
    return LLR