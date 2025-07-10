# Enhancing Packet Loss Concealment with Generative Spectrogram Inpainting

This repository contains the project files to replicate the experiments described in the technical report provided.

Below are links to the datasets:
* [Medley Solos DB](https://zenodo.org/records/3464194)
* [GoodSounds](https://www.upf.edu/web/mtg/good-sounds)
  
While to generate the synthesized sequences, MIDI files were taken from the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro#v300) collection and wav files were synthesized with FluidSynth and the [GeneralUser GS soundfont](https://schristiancollins.com/generaluser.php), for a total of 45 hours.

To start train the _bin2bin_ network from scratch, set the appropriate dataset path variable(s) in `train_config.yaml`, then run `train_b2b_lpc.py` while passing the chosen dataset name as an argument. 

To start generate inpainted files, set the appropriate paths in `fw_config.yaml`, then run `inpaint_b2b_lpc.py`.

A pretrained Generator checkpoint is available [here](https://mega.nz/file/XdYgWLoS#q9rfU4ZsTp5QfnpRuWyVdBfBAQMLXkMc0hpehZt_MJU);

For further informations, please contact me at c.aironi@staff.univpm.it
