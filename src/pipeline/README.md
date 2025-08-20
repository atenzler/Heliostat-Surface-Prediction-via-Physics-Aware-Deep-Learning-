# Pipeline

This folder contains the core training, hyperparameter optimization, 
and evaluation pipeline for heliostat surface prediction with DeepLARTS.  

The code here integrates:
- Preprocessed flux/surface data
- The DeepLARTS neural network model
- The ARTIST raytracer for flux validation
- Bayesian hyperparameter optimization with Optuna

> **Note:** Several scripts contain absolute file paths with references
to the correct files, but must be adapted to your local or cluster 
environment before running.

## Files

- **run_experiment.py**  
  Main entry point for training.  
  Supports both:
  - **Local single-GPU runs** (for debugging and small experiments).
  - **Multi-GPU distributed training on JUWELS** (up to 4 GPUs) via 
    `torch.distributed` (NCCL backend).  

  Hyperparameters, penalties, and experiment settings are configured 
  at the top of the script or via imported configs.

- **impl.py**  
  Core training and evaluation logic.  
  Contains:
  - Custom loss functions (RMSE, SSIM, curvature, edge dip, z-range, etc.)
  - Training loop with early stopping
  - Evaluation on validation/test sets
  - Logging utilities for TensorBoard (heatmaps, 3D surfaces)

- **func.py**  
  Utility functions for the pipeline:
  - Dataset splitting
  - Folder structure creation (cluster/local)
  - Logging setup and summaries
  - Loss plotting and simple metrics

- **data_setup.py**  
  Functions to set up:
  - Augmented and ideal surfaces
  - ARTIST scenarios and prototype surfaces
  - Data splits for different experimental settings (close to receiver, 
    high-ray, one-sun-position)
  - Model builders with/without canting inputs

- **dataset.py**  
  PyTorch `Dataset` classes:
  - **HeliostatChunkDataset** – training dataset from chunked flux images
  - **HeliostatTestDataset** – test wrapper  
  Supports optional facet canting vector inputs.

- **integrate_raytracer.py**  
  Bridge between NN predictions and ARTIST raytracer:  
  - Overwrites scenario surfaces with NN-predicted Z-residuals
  - Runs heliostat-based raytracing for given sun positions
  - Returns flux maps for comparison with ground truth

- **bayes_optuna.py**  
  Bayesian hyperparameter optimization using Optuna.  
  - Samples penalty weights (z-constraint, curvature, edge dip, tilt, z-range)  
  - Runs multiple trials on 1 GPU  
  - Logs results to TensorBoard and a SQLite database  
  - Saves best trial configuration as JSON  

- **test_model.py**  
  Evaluation script for trained models.  
  - Loads trained checkpoints (best surface / best train model)  
  - Reconstructs predicted surfaces from test dataset  
  - Evaluates with ARTIST scenario and logs metrics  
  - Produces plots (3D reconstructions, surface heatmaps)  

