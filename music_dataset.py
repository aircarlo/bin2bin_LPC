import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import librosa
from lpc import My_LPC_model


class Music_dataset_raw(Dataset):
    """
    Dataset class for raw spectrograms (complex valued).
    """

    def __init__(self, wav_list, params, mode, device):
        self.wav_list = wav_list # full path list
        self.par = params
        self.mode = mode
        self.device = device

        # STFT class
        self.spectrogram = T.Spectrogram(n_fft=self.par["FFT_SIZE"],
                                         hop_length=self.par["FFT_HOP"],
                                         power=None,
                                         normalized=False) # .to(self.device)

        # LPC class
        self.lp_model = My_LPC_model(p=self.par["LPC_ORD"], taper=False)


    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        # load the clean waveform @ 44100, mono (Librosa)
        clean_waveform, orig_sr = librosa.load(self.wav_list[idx], sr=self.par["WORKING_SR"], mono=True)

        # trim to n*GAP_SIZE
        len_samples = clean_waveform.shape[0]        
        clean_waveform = clean_waveform[0:(len_samples // self.par["GAP_SIZE"] * self.par["GAP_SIZE"])]

        lossy_waveform = np.copy(clean_waveform)

        if self.mode == 'val':
            start_bin = 0
        elif self.mode=='train':
            start_bin = np.random.randint(0, lossy_waveform.shape[0] - self.par["CONTEXT"])
        end_bin = start_bin + self.par["CONTEXT"]
        
        t_clean_frame = clean_waveform[start_bin:end_bin]
        # LPC prediction
        pred = self.lp_model.predict(ctxt=t_clean_frame[:-2*self.par["GAP_SIZE"]], samp=2*self.par["GAP_SIZE"])
        t_lossy_frame = np.concatenate((lossy_waveform[start_bin:end_bin-2*self.par["GAP_SIZE"]], pred))

        # compute the spectrograms
        S_clean_frame = self.spectrogram(torch.from_numpy(t_clean_frame).unsqueeze(0).float())
        S_lossy_frame = self.spectrogram(torch.from_numpy(t_lossy_frame).unsqueeze(0).float())
        
        # mod/pha
        S_clean_frame_ABS = torch.abs(S_clean_frame)**0.5
        S_lossy_frame_ABS = torch.abs(S_lossy_frame)**0.5
        S_clean_frame_PHA = torch.angle(S_clean_frame)
        S_lossy_frame_PHA = torch.angle(S_lossy_frame)

        # T-F normalizations
        S_clean_frame_ABS /= 3.5
        S_lossy_frame_ABS /= 3.5

        return S_lossy_frame_ABS, S_clean_frame_ABS, S_lossy_frame_PHA, torch.from_numpy(t_clean_frame).unsqueeze(0).float()




if __name__ == "__main__":
    pass
