# Data

This folder contains all datasets used in the thesis.  

Data is divided into:
- **raw/** → input for preprocessing scripts (`src/preprocessing/`)
- **processed/** → cleaned and prepared data used directly in the training and evaluation pipeline (`src/pipeline/`)

## Structure

- **raw/**  
  Contains original data sources used for preprocessing. Examples:
  - Measured heliostat surfaces (`all_heliostat_surfaces.json`)
  - Deflectometry datasets
  - Scenario files for ARTIST raytracer (`.h5`)
  - Canting property files

  These files are required if you want to **re-run preprocessing** 
  to regenerate training data.

- **processed/**  
  Contains data outputs from preprocessing, ready for use in the 
  pipeline. Examples:
  - `surfaces/` – ideal surfaces, z-displacements, augmented surfaces  
  - `flux_images/` – flux density images, sun vectors, and metadata  
  - `chunks/` – chunked flux/surface datasets for PyTorch `Dataset` classes  

  These files are the **starting point for training and evaluation**.  
  Running the demo pipeline requires only the processed data, not the raw inputs.

## Notes

- Some datasets are reduced in size to allow quick testing 
- For full reproducibility of all experiments, both raw and processed 
  data must be available.  
-
