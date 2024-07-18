import os, sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio.transforms as T
from tqdm import tqdm
from datetime import datetime
from loss import *
from utils import save_checkpoint, save_gen_specs, file_parse
from music_dataset import Music_dataset_raw
from discriminator_model import Discriminator
from generator_model import Generator


# parse parameters
with open(r'train_config.yaml') as file:
    par = yaml.load(file, Loader=yaml.FullLoader)
par["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
if par["SAVE_IMG_AT"]=='all':
    par["SAVE_IMG_AT"] = list(range(1,par["NUM_EPOCHS"]+1))
data = sys.argv[1]
if data not in ['medley_solos', 'good_sounds', 'synth_sounds']:
    raise Exception('Please specify dataset: medley_solos, good_sounds or synth_sounds')

# ISTFT class
inv_spec = T.InverseSpectrogram(n_fft=par["FFT_SIZE"], hop_length=par["FFT_HOP"]).to(par["DEVICE"])

# step counter (used for logs)
step_cnt = 0

def train_fn(disc, gen, loader, opt_disc, opt_gen, loss, lr_schedulers, tb_writer):
    global step_cnt
    loop = tqdm(loader, leave=True)
    D_loss_epoch = 0.0
    SC_loss_epoch = 0.0
    MAG_loss_epoch = 0.0
    G_loss_epoch = 0.0

    for idx, (x, y, x_pha, _) in enumerate(loop):
        x = x.to(par["DEVICE"]) # lossy sample
        y = y.to(par["DEVICE"]) # clean sample
        x_pha = x_pha.to(par["DEVICE"]) # lossy sample (phase)

        # Train the discriminator
        opt_disc.zero_grad()
        y_fake = gen(x)
        D_real = disc(x, y)
        D_real_loss = loss[0](D_real, torch.ones_like(D_real)) # 1 = "real"
        D_fake = disc(x, y_fake.detach())
        D_fake_loss = loss[0](D_fake, torch.zeros_like(D_fake)) # 0 = "fake"
        D_loss = (D_real_loss + D_fake_loss) / 2
        D_loss_epoch += D_loss.item()

        if idx % par["N_CRITICS"] == 0: # update discriminator every n_critics iterations
            D_loss.backward()
        opt_disc.step()

        # Train the generator
        opt_gen.zero_grad()
        D_fake = disc(x, y_fake)
        G_fake_loss = loss[0](D_fake, torch.ones_like(D_fake))
        SC_loss = loss[1](y_fake, y) # * par["LAMBDA"]
        MAG_loss = loss[2](y_fake, y) # * par["LAMBDA"]

        G_loss = G_fake_loss + SC_loss + MAG_loss

        SC_loss_epoch += SC_loss.item()
        MAG_loss_epoch += MAG_loss.item()
        G_loss_epoch += G_loss.item()

        if torch.isnan(G_loss).any():
            print('G Loss NaN! Experiment stopped.')
            sys.exit()

        # update generator every iteration
        G_loss.backward()
        opt_gen.step()

        # ######## early stop for debug
        # if idx==50:
        #     return 0,0,0,0,[0,0]
        # ########

    lr_schedulers[0].step()
    lr_schedulers[1].step()
    lr_values = (opt_disc.param_groups[0]["lr"], opt_gen.param_groups[0]["lr"])

    return D_loss_epoch, G_loss_epoch, SC_loss_epoch, MAG_loss_epoch, lr_values


def main(data):

    # create log folder
    if not os.path.exists(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"])):
        os.makedirs(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"]))
        os.makedirs(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["IMG_EVAL_DIR"]))
        os.makedirs(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"]))
    else:
        raise Exception(f'Log folder {os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"])} already exists!')
    
    now_tstamp = datetime.now()

    # dump yaml param file to txt
    log_info = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], "params.txt")
    with open(log_info, 'w') as f:
        f.write(f'Experiment {par["EXPERIMENT_ID"]} started at {now_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}\n\n')
        for k in par.keys():
            f.write(f'{k}: {par[k]}\n')

    # initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"]))

    # define network models
    disc = Discriminator(in_channels=1, k_size=par["D_KERNEL_SIZE"]).to(par["DEVICE"])
    gen = Generator(in_channels=1, features=par["G_FEATURES"]).to(par["DEVICE"])

    # define optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=par["D_LEARNING_RATE"], betas=(0.9, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=par["G_LEARNING_RATE"], betas=(0.9, 0.999))

    # learning rate schedulers
    opt_disc_scheduler = CosineAnnealingLR(opt_disc,
                                           T_max = par["NUM_EPOCHS"], # Maximum number of iterations.
                                           eta_min = 0.0001) # Minimum learning rate.
    opt_gen_scheduler = CosineAnnealingLR(opt_gen,
                                          T_max = par["NUM_EPOCHS"], # Maximum number of iterations.
                                          eta_min = 0.0001) # Minimum learning rate.

    # list of available loss fn
    adv_loss = nn.MSELoss()                 # LSGAN -> MSELoss
    td_loss = nn.L1Loss()                   # time domain loss
    mag_loss = LogSTFTMagnitudeLoss()       # reconstruction loss
    sc_loss = LogSpectralConvLoss()         # reconstruction loss

    # define dataset and dataloader
    if data == 'medley_solos':
        f_list = file_parse(par["MEDLEY_SOLOS_PATH"], ext='wav', return_fullpath=True)
    if data == 'good_sounds':
        f_list = file_parse(par["GOOD_SOUNDS_PATH"], ext='wav', return_fullpath=True)
    if data == 'synth_sounds':
        f_list = file_parse(par["SYNTH_SOUNDS_PATH"], substr = 'segmented', ext='wav', return_fullpath=True)

    train_dataset = Music_dataset_raw(f_list,
                                      par,
                                      mode='train',
                                      device = par['DEVICE'])
    
    train_loader = DataLoader(train_dataset,
                              batch_size=par["BATCH_SIZE"],
                              shuffle=True,
                              num_workers=par["NUM_WORKERS"])
    
    val_dataset = Music_dataset_raw(f_list,
                                    par,
                                    mode='val',
                                    device = par['DEVICE'])

    ex_data = val_dataset[10]  # pick a sample for plot

    print(f'Start training experiment {par["EXPERIMENT_ID"]}')
    start_tstamp = datetime.now()

    for epoch in range(par["NUM_EPOCHS"]):
        print(f'\nEPOCH: {epoch+1}/{par["NUM_EPOCHS"]}')

        train_results = train_fn(disc,
                                 gen,
                                 train_loader,
                                 opt_disc,
                                 opt_gen,
                                 [adv_loss, sc_loss, mag_loss, td_loss],
                                 [opt_disc_scheduler, opt_gen_scheduler],
                                 writer)

        # Tensorboard logs
        writer.add_scalar('PER_EPOCH/D_ADV_LOSS',             (train_results[0]/len(train_loader)),      epoch) # Discriminiator MSE loss
        writer.add_scalar('PER_EPOCH/G_ADV_LOSS',             (train_results[1]/len(train_loader)),      epoch) # Generator MSE loss
        writer.add_scalar('PER_EPOCH/SC_LOSS',                (train_results[2]/len(train_loader)),      epoch) # Spectral Convergence loss
        writer.add_scalar('PER_EPOCH/MAG_LOSS',               (train_results[3]/len(train_loader)),      epoch) # Magnitude STFT loss
        writer.add_scalar('PER_EPOCH/OPT_DISC',               (train_results[4][0]),                     epoch)
        writer.add_scalar('PER_EPOCH/OPT_GEN',                (train_results[4][1]),                     epoch)
        writer.flush()

        # checkpoint
        if epoch+1 in par["SAVE_MODEL_AT"]:
            save_checkpoint(gen, opt_gen, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_gen_e_' + str(epoch+1) + '.pth'))
            # do not save discriminator                                                                                                                                                                         
            # save_checkpoint(disc, opt_disc, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_disc_e_' + str(epoch) + '.pth'))

        # save last epock checkpoint (overwrite)
        save_checkpoint(gen, opt_gen, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_gen_LAST.pth'))

        if epoch+1 in par["SAVE_IMG_AT"]:
            
            save_gen_specs(gen, ex_data, epoch, par)

    writer.close()
    end_tstamp = datetime.now()
    print('DONE')
    print(f'Training started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Training finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')


if __name__ == "__main__":
    main(data)
