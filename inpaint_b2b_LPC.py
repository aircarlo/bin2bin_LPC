import os, sys
import yaml
import numpy as np
import torch
import torchaudio.transforms as T
import librosa
import soundfile as sf
from generator_model import Generator
from utils import file_parse, m_a
from lpc import My_LPC_model


# parse parameters
with open(r'fw_config.yaml') as file:
    par = yaml.load(file, Loader=yaml.FullLoader)

# create log folder
if not os.path.exists(par["DEST_PATH"]):
    os.makedirs(par["DEST_PATH"])
else:
    raise Exception(f'Destination folder {par["DEST_PATH"]} already exists!')

# cross-fade profiles
xfade1_in = np.hanning(2*par["XF1"])[:par["XF1"]]
xfade1_out = np.hanning(2*par["XF1"])[par["XF1"]:]
xfade2_in = np.hanning(2*par["XF2"])[:par["XF2"]]
xfade2_out = np.hanning(2*par["XF2"])[par["XF2"]:]

lpc_model = My_LPC_model(p=par["LP_ORD"], taper=False)

# STFT class
spec = T.Spectrogram(n_fft=par["FFT_SIZE"],
                     hop_length=par["FFT_HOP"],
                     power=None,
                     normalized=False)

# ISTFT class
inv_spec = T.InverseSpectrogram(n_fft=par["FFT_SIZE"],
                                hop_length=par["FFT_HOP"])

bin2bin = Generator(features=par["G_FEATURES"]).to(par["DEVICE"])
checkpoint = torch.load(par["B2B_CKPT_FILE"], map_location=par["DEVICE"])
bin2bin.load_state_dict(checkpoint["state_dict"])
bin2bin.eval()
print(f'>> bin2bin checkpoint loaded')

f_list = file_parse(par["DATA_PATH"], ext='wav', return_fullpath=True)

for w_idx, wav_fname in enumerate(f_list):
    try:
        mask_fname = os.path.join(par["TRACE_PATH"], os.path.split(wav_fname)[-1][:-3] + 'txt')
        print(f'{w_idx+1}/{len(f_list)} - Inpaint {os.path.split(wav_fname)[-1]}...', end='')

        lossy_waveform, _ = librosa.load(wav_fname, sr=par["WORK_SR"], mono=True)

        # trim a un multiplo di 512 campioni
        len_samples = lossy_waveform.shape[0]
        lossy_waveform = lossy_waveform[0:(len_samples // par["GAP_SIZE"] * par["GAP_SIZE"])]

        # read gap mask from file
        with open(mask_fname, 'r') as f:
            frame_mask = f.read().split('\n')
        # transform to list of int and invert: from [0=non-lost, 1=lost] to [0=lost, 1=non-lost]
        frame_mask = [-int(i)+1 for i in frame_mask[:-1]]

        # generate the gap mask at sample level
        sample_mask = np.repeat(frame_mask, par["GAP_SIZE"])

        # append a zero-head
        lossy_waveform_ext = np.zeros(lossy_waveform.shape[0]+par["CONTEXT"]-par["GAP_SIZE"])
        lossy_waveform_ext[-lossy_waveform.shape[0]:] = lossy_waveform

        # funzione di inpaint 
        inpainted_waveform_ext = np.copy(lossy_waveform_ext)
        prev_lost = False

        for k, i in enumerate(range(0, lossy_waveform_ext.shape[0]-par["CONTEXT"], par["GAP_SIZE"])):
            start_s = i
            end_s = i+par["CONTEXT"]
            if frame_mask[k] == 0: # if lost
                # LPC prediction
                pred_LPC = lpc_model.predict(ctxt=inpainted_waveform_ext[start_s:end_s-par["GAP_SIZE"]], samp=par["GAP_SIZE"]+par["XF2"], recompute_c=not(prev_lost))
                # bin2bin refinement
                t_lossy_frame = np.concatenate((inpainted_waveform_ext[start_s:end_s-(par["GAP_SIZE"]+par["XF2"])], 0.8*pred_LPC))
                S_lossy_frame = spec(torch.from_numpy(t_lossy_frame).unsqueeze(0).float())
                # abs/pha
                S_lossy_frame_ABS = torch.abs(S_lossy_frame)**0.5
                S_lossy_frame_PHA = torch.angle(S_lossy_frame)
                # T-F normalizations
                S_lossy_frame_ABS /= 3.5
                # bin2bin forward
                S_lossy_frame_ABS = S_lossy_frame_ABS.to(par["DEVICE"])
                with torch.no_grad():
                    S_inp = bin2bin(S_lossy_frame_ABS.unsqueeze(0))
                S_inp = S_inp.detach().cpu()
                # back to time-domain
                S_inp_cplx = ((S_inp*3.5)**2) * torch.exp(1j*S_lossy_frame_PHA)
                t_inp = np.array(inv_spec(S_inp_cplx.squeeze(1), length=par["CONTEXT"]))
                pred_b2b = t_inp[0,-(par["GAP_SIZE"]+par["XF2"]):]
                # xfade intra-gap
                pred_b2b[0:par["XF1"]] *= xfade1_in
                pred_b2b[0:par["XF1"]] += pred_LPC[0:par["XF1"]] * xfade1_out
                # reassemble
                inpainted_waveform_ext[end_s-par["GAP_SIZE"]:end_s] = pred_b2b[0:par["GAP_SIZE"]]
                prev_lost = True
            
            else:
                if prev_lost:
                    # xfade after gap
                    inpainted_waveform_ext[end_s-par["GAP_SIZE"]:end_s-par["GAP_SIZE"]+par["XF2"]] *= xfade2_in
                    inpainted_waveform_ext[end_s-par["GAP_SIZE"]:end_s-par["GAP_SIZE"]+par["XF2"]] += (pred_b2b[-par["XF2"]:] * xfade2_out)
                    prev_lost = False

        # remove zero-head
        inpainted_waveform = inpainted_waveform_ext[par["CONTEXT"]-par["GAP_SIZE"]:]

        # save as WAV
        i_file = os.path.join(par["DEST_PATH"], os.path.split(wav_fname)[-1])
        sf.write(i_file, inpainted_waveform, par["WORK_SR"])

        print('done')
    except:
        print('ERROR')
