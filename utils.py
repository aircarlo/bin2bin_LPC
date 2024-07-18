import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


# parse files with generic extensions
def file_parse(root_dir, ext='', substr='', return_fullpath=True):
    file_list = []
    print(f'parse {root_dir}... ', end='')
    for root, _, files in os.walk(root_dir, topdown=False):
        for name in files:
            if os.path.isfile(os.path.join(root, name)) and name[-3:] == ext and substr in root:
                if return_fullpath:
                    file_list.append(os.path.join(root, name))
                else:
                    file_list.append(name)
    print(f'found {len(file_list)} {ext} files')
    return file_list


# save generator output as img
def save_gen_specs(gen, sample, epoch, params, scale_dB=False):
    folder = os.path.join(params["LOG_PATH"], params["EXPERIMENT_ID"], params["IMG_EVAL_DIR"])
    x, y, *_ = sample
    x = x.unsqueeze(0).to(params["DEVICE"])
    y = y.unsqueeze(0).to(params["DEVICE"])

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        if scale_dB:
            y_fake = F.amplitude_to_DB(torch.abs(y_fake)**2, multiplier=10, amin=1e-6, db_multiplier=10)
        plt.imshow(y_fake[0][0].detach().cpu().numpy(), origin='lower')
        plt.savefig(folder + f"/y_gen_{epoch}.png")
        plt.close()

        if epoch == 0: # at epoch 0 save also input (with gap) and target (clean)
            if scale_dB:
                x = F.amplitude_to_DB(torch.abs(x)**2, multiplier=10, amin=1e-6, db_multiplier=10)
            plt.imshow(x[0][0].detach().cpu().numpy(), origin='lower')
            plt.savefig(folder + "/input.png")
            plt.close()
            if scale_dB:
                y = F.amplitude_to_DB(torch.abs(y)**2, multiplier=10, amin=1e-6, db_multiplier=10)
            plt.imshow(y[0][0].detach().cpu().numpy(), origin='lower')
            plt.savefig(folder + "/target.png")
            plt.close()

    print('>> examples saved')


def save_checkpoint(model, optimizer, filepath):
    print(">> checkpoint saved")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(checkpoint_file, model, device, optimizer=None, lr=None):
    print(f'>> loading checkpoint {checkpoint_file}... ', end='')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    print('done')
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_rate_burst(mask_file):
    """
    Read a mask file and return the loss rate and the maximum burst of consecutive lost packets
    parameters
    mask file: full path txt file, where 0 = non-lost, 1 = lost
    """
    with open(mask_file, 'r') as f:
        flags = f.read().split('\n')
    flags = [int(i) for i in flags[:-1]]
    burst = max(map(len, ''.join(map(str, flags)).split('0')))
    rate = sum(flags)/len(flags)
    return rate, burst


# moving average filter
def m_a(signal, window_size):
    filtered_signal = []
    window = []
    for i in range(len(signal)):
        window.append(signal[i])
        if len(window) > window_size:
            window.pop(0)
        filtered_signal.append(sum(window) / len(window))
    
    return np.array(filtered_signal)

if __name__ == "__main__":
    pass