# Enhancing Packet Loss Concealment with Generative Spectrogram Inpainting

This repository contains all the project files to replicate the experiments described in the technical reports provided.

To start train the _bin2bin_ network from scratch, set the appropriate dataset path variable(s) in `train_config.yaml`, along with the selection of _full_ or _lite_ architecture, then run `train_b2b_lpc.py` while passing the chosen dataset name as an argument. 

To start generate inpainted files, set the appropriate paths in `fw_config.yaml`, then run `inpaint_b2b_lpc.py`.

Pretrained Generator checkpoints for both _lite_ and _full_ bin2bin versions are available [here](https://mega.nz/folder/mA4kQYCZ#mTp8urMkT-vlGDtZSwEjpA);

For further help, please contact me at c.aironi@staff.univpm.it
