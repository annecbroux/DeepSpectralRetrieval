# DeepSpectralRetrieval
This repo contains the codes for the deep learning-based retrieval of snowfall microphysics from dual-frequency spectral radar measurements.
The retrieval is described in the corresponding publication: Billault-Roux et al., 2023, Atmospheric Measurement Techniques, https://doi.org/10.5194/amt-16-911-2023.

Questions should be directed at anne-claire.billault-roux@epfl.ch

### Structure
It consists of three directories:
* **`spectra_database_creation`**: scripts to generate the synthetic training set (used for training the decoder part of the model). 
* **`decoder`**: scripts to train the decoder part of the model (once training set is generated)
* **`encoder`** : script to train the encoder part of the model. Note that this requires to have pre-processed files containing dual-frequency Doppler spectral radar measurements remapped to a common grid.

### Requirements
#### Generation of the training set with a radiative transfer model
This requires a preliminary installation of PAMTRA (https://pamtra.readthedocs.io/en/latest/). \
Note that to this date, PAMTRA does **not** run in conda environments: scripts in the **spectra_database_creation** directory should therefore not be run with an active conda environment.

#### Deep learning framework
For this part however, we recommend that the user sets up a conda environment to use / run the codes of the deep learning framework (i.e. in the **decoder** and **encoder** directories). \
python3 is required (python3.7 to 3.9 were tested) , with at least the following packages:
```
pytorch>=1.9
tensorboard>=2.6 (see e.g. https://pytorch.org/docs/stable/tensorboard.html )
scipy>=1.5.3
pandas>=1.3.5
h5py>=3.3
numpy>=1.18
matplotlib>=3.4
tqdm
```
Examples of data on which to run the code were not included (yet) because quite heavy, but can be shared upon request.
