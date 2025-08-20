Thesis Repository
===========================

This repository contains the code, data, and experiments used for my 
master thesis on heliostat surface prediction with DeepLARTS, 
a deep learning framework adapted from Lewen et al. (2024).

The repository is organized to allow:
- Reproducibility of a demo training/evaluation run
- Access to preprocessing scripts used to prepare the dataset
- Integration with the ARTIST raytracer
- Hyperparameter optimization with Optuna
- Evaluation of trained models on test datasets

------------------------------------------------------------
Repository structure
------------------------------------------------------------

'''

data/             Raw + processed data
  ├── raw/        Inputs for preprocessing
  └── processed/  Data used directly in training/evaluation

external/         External dependencies
  └── ARTIST_modified/   Adapted ARTIST raytracer version

src/              Source code
  ├── preprocessing/     Scripts for generating processed data
  ├── models/            NN model definitions (DeepLARTS, StyleGAN2)
  ├── pipeline/          Training, evaluation, raytracer integration
  ├── hpo/               Hyperparameter optimization (Optuna)
  ├── evaluation/        Evaluation scripts
  └── config/            Configuration (my_cfg.py etc.)

'''

------------------------------------------------------------
Notes
------------------------------------------------------------

- File paths in some scripts are currently hardcoded and must be adapted 
  to your local or cluster environment.

- Preprocessing is not required for reproducing the demo run 
  (only processed data is needed).

- The ARTIST raytracer is included in external/ARTIST_modified/.
  It is an external tool and not developed in this thesis, but 
  slight modifications were made for compatibility.
