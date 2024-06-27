# MDSM
Based on code from [MDSM](https://github.com/zengyi-li/MDSM) for reproducing results in [Multiscale Denoising Score Matching](https://arxiv.org/abs/1910.07762)

Adapted for downsampled magnitude images from the FastMRI brain dataset.

MDSM trains a neural network as an Energy-Based Model (EBM) using denoising score matching. The resulting energy model can be sampled using Annealed Langevin Dynamics.

## Requirements
 * pyTorch
 * torchvision