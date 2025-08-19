# Preprocessing

This folder contains all scripts used to prepare heliostat surface and flux data
before training neural networks. These steps are not strictly necessary for
running the demo pipeline (which uses already-processed data), but they are an
integral part of the thesis methodology and included here for completeness.

## Structure

- **augment_surfaces.py**  
  Generates augmented training surfaces by rotating and interpolating
  real Z-displacement surfaces. Saves results as `.pt` or `.json`.

- **classify_heliostats.py**  
  Classifies heliostats into *receiver-canting* and *other* types by reading
  their JSON property files. Optionally plots heliostat positions in local ENU
  coordinates.

- **generate_ideal_heliostats.py**  
  Generates "ideal" heliostat NURBS surfaces from facet canting parameters.
  Each heliostat is reconstructed and saved as a `.pt` tensor and a `.png`
  visualization.

- **compute_z_displacements.py**  
  Computes Z-displacement tensors between real and ideal surfaces.  
  Aligns facets, subtracts ideal from real, and saves results as `.pt`.

- **heliostat_positions_utils.py**  
  Utilities for loading real heliostat positions from deflectometry datasets.

- **compute_nurbs_without_canting.py**  
  Removes canting effects from NURBS surfaces, comparing measured control
  points against ideal surfaces. Saves surfaces "without canting" into JSON.

- **sun_positions_utils.py**  
  Utilities for computing solar azimuth/elevation angles and converting them
  into 3D/4D vectors. Includes random sampling functions and equidistant
  position generation.

- **generate_flux_images.py**  
  Runs the **ARTIST raytracer** to generate flux density images for augmented
  heliostat surfaces.  
  Steps:
  1. Load a scenario definition (`.h5`).
  2. Load augmented surfaces and ideal heliostat surfaces.
  3. Randomly assign heliostat positions (simulated or real) and canting vectors.
  4. Overwrite the scenario with these surfaces/positions.
  5. Raytrace flux images for multiple random sun positions.
  6. Save flux images, sun vectors, and metadata in chunked files (`.pt` + `.json`).  

  These flux images form the **supervised training targets** for the neural network.

## Notes

- All scripts currently contain **incomplete paths**.
  Replace them with your own paths or adapt to the local dataset structure.
- These scripts are **not required** to reproduce the provided demo run,
  but document the full preprocessing pipeline used in the thesis.
- Outputs of these scripts populate:
  - `data/processed/` (used in training/demo run)
  - `data/raw/` (original JSONs and measurement data)
