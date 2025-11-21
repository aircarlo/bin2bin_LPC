import torch
import torch.nn as nn
import torchaudio

class SpectralConvLoss(nn.Module):
    """
    Spectral convergence loss module.
    see https://arxiv.org/pdf/1808.06719.pdf
    """

    def __init__(self):
        super(SpectralConvLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm((y_mag - x_mag), p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """
    Log STFT magnitude loss module.
    see https://arxiv.org/pdf/1808.06719.pdf
    """

    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        x_mag = torch.clamp(x_mag, min=1e-6, max=1e3) # avoid underflows & overflows
        y_mag = torch.clamp(y_mag, min=1e-6, max=1e3)
        return F.l1_loss(torch.log(x_mag), torch.log(y_mag))
