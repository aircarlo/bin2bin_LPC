import numpy as np
import scipy.signal
import scipy.linalg
import librosa


class My_LPC_model:
    
    def __init__(self, p, taper=False): # taper: Bool - smooth the context with an Hanning window, before computing the coefficients
        self.p = p
        self.taper = taper
        self.coeff = None

    def autocorr(self, seq, p):
        R = [seq[0:len(seq)-k] @ seq[k:len(seq)] for k in range(p)]
        return R

    def compute_coeff(self, seq):
        if self.taper:
            seq = seq*np.hanning(len(seq))
        R = self.autocorr(seq, self.p) # R00 = [R(0)...R(p-1)]
        R[0] += 1e-3
        b = self.autocorr(seq, self.p+1)[1:] # b = [R(1)...R(p)]'
        LPC = scipy.linalg.solve_toeplitz((R,R),b)
        return LPC

    def predict(self, ctxt, samp, recompute_c=True): # recompute_c: Bool - whether to recompute the coefficients
        if recompute_c:
            self.coeff = self.compute_coeff(ctxt)
        prediction = np.zeros(samp)
        ctxt = ctxt[-self.p:]

        for i in range(samp):
            prediction[i] = np.dot(ctxt, self.coeff[::-1])
            ctxt = np.roll(ctxt, -1)
            ctxt[-1] = prediction[i]

        return prediction