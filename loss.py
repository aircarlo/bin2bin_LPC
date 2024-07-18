import torch
import torch.nn as nn
import torchaudio

class LogSpectralConvLoss(nn.Module):
    """
    log-Spectral convergence loss module.
    """

    def __init__(self):
        super(LogSpectralConvLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm((y_mag - x_mag), p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    """
    Log STFT magnitude loss module.
    """

    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return nn.functional.l1_loss(torch.log(y_mag+1e-8), torch.log(x_mag+1e-8))
