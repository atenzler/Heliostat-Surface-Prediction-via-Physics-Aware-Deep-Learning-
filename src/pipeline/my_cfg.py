"""
my_cfg.py

Configuration module for DeepLARTS, adapted from Lewen et al. (2024).

Uses `yacs.CfgNode` for hierarchical configuration management.
Defines directory paths, training constants, and DeepLARTS parameters.

Sections:
- DIRECTORIES: Local and cluster directories
- DEEPLARTS.TRAIN: Training-related constants
- (extendable for VALID, TEST, etc.)

Note:

Adapt paths to your environment before running.
"""


from yacs.config import CfgNode as CN


_C = CN()
# UNIQUE EXPERIMENT IDENTIFIER
_C.DIRECTORIES = CN()
_C.DIRECTORIES.LOCAL = CN()
_C.DIRECTORIES.LOCAL.HOMEDIR = r'...'
_C.DIRECTORIES.LOCAL.MODELDIR = r'...'


_C.DEEPLARTS = CN()
_C.DEEPLARTS.TRAIN = CN()
_C.DEEPLARTS.TRAIN.TARGET_IMAGE_SIZE = 64
_C.DEEPLARTS.TRAIN.SURFACE_MIN = -0.004
_C.DEEPLARTS.TRAIN.SURFACE_MAX = 0.004
_C.DEEPLARTS.TRAIN.NSUNPOS = 8