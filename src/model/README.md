# Models

This folder contains all neural network–related code used in the thesis.
The architectures are based on Lewen et al. (2024)
([arXiv:2408.10802](https://doi.org/10.48550/arXiv.2408.10802)), with
slight modifications introduced for integration with the pipeline.

## Files

- **Parameter_Netzwerk.py**  
  Defines all parameter dictionaries for model architecture,
  encoders, StyleGAN2, training, testing, and data settings.
  These are passed to `init_deepLarts` to build or load a model.

- **my_deepLarts.py**  
  Main DeepLARTS model definition and initialization.
  Includes:
  - Utility functions for saving/loading models
  - Encoders (AdaIN-based, weight-demodulated)
  - Style mapping from scalar inputs
  - `StyleDeepLarts` model (encoder + StyleGAN2 generator)
  - `init_deepLarts` for setting up new or pretrained models

- **styleGAN2_surfaces.py**  
  Modified implementation of StyleGAN2 for surface generation,
  adapted from [lucidrains/stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch).
  Adjusted for:
  - single-channel (grayscale) surface data
  - integration with the pipeline
  - latent space interpolation, projection, and FID evaluation

- **utils.py**  
  Utility functions for plotting surfaces, adjusting inputs,
  reproducibility (seeding), and cluster-safe logging.

## Notes

- The core design follows Lewen et al. (2024), with minor adjustments.
- StyleGAN2 code is redistributed and modified for reproducibility.
- All models are integrated into the training pipeline in
  `src/pipeline/`.

