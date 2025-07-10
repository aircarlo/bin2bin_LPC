import numpy as np
import scipy.signal
import scipy.linalg


class LPC_model_fast: 
    
    def __init__(self, p, taper=False): # taper: Bool - smooth the context with an Hanning window, before computing the coefficients
        self.p = p
        self.taper = taper
        self.coeff = None

    def autocorr(self, seq, maxlag):
        N = len(seq)
        result = np.correlate(seq, seq, mode='full')
        mid = len(result) // 2
        return result[mid:mid+maxlag+1]

    def compute_coeff(self, seq):
        if self.taper:
            seq = seq * np.hanning(len(seq))
        R_full = self.autocorr(seq, self.p)
        R_full[0] += 1e-3
        R = R_full[:self.p]  # autocorrelazione R(0)...R(p-1)
        b = R_full[1:self.p+1]  # R(1)...R(p)
        LPC = scipy.linalg.solve_toeplitz((R, R), b)
        return LPC

    def predict(self, ctxt, samp, recompute_c=True):
        if recompute_c:
            self.coeff = self.compute_coeff(ctxt)
        prediction = np.zeros(samp)
        buffer = ctxt[-self.p:].copy()

        for i in range(samp):
            prediction[i] = np.dot(self.coeff[::-1], buffer)
            buffer[:-1] = buffer[1:]
            buffer[-1] = prediction[i]

        return prediction
        