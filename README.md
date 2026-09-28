# Adaptive Multiscale Operator Correction (AMOC)

PyTorch implementation of **Adaptive Multiscale Operator Correction (AMOC)** for PDE solution correction using a neural-operator prediction followed by physics-informed correction in a sparse spectral representation.

This repository contains code for experiments on:
- Darcy flow
- Heat equation
- 2D Poisson equation
- 3D Poisson equation
- 3D Helmholtz equation

## Overview

AMOC is a three-stage pipeline:
1. Train a neural operator to produce a coarse PDE solution.
2. Project the prediction onto a sparse spectral basis using wavelet or Fourier modes.
3. Perform physics-informed refinement in the selected spectral representation.

Two correction variants are supported:
- **W-AMOC**: localized wavelet-based correction
- **F-AMOC**: Fourier-mode-based correction




The implementation builds on ideas and software ecosystems associated with neural operators, operator learning, and physics-informed learning discussed in the manuscript.

### Fourier Neural Operator (FNO)

```bibtex
@inproceedings{
li2021_fno,
title={Fourier Neural Operator for Parametric Partial Differential Equations},
author={Zongyi Li and Nikola Borislavov Kovachki and Kamyar Azizzadenesheli and Burigede liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=c8P9NQVtmnO}
}
```

```bibtex
@article{PANDEY2026108860,
title = {An efficient wavelet-based physics-informed neural network for multiscale problems},
journal = {Neural Networks},
volume = {200},
pages = {108860},
year = {2026},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2026.108860},
url = {https://www.sciencedirect.com/science/article/pii/S0893608026003217},
author = {Himanshu Pandey and Anshima Singh and Ratikanta Behera}
}
```
- https://drive.google.com/drive/folders/1itTmOVgtIAs53xGvSjgPpoQ9zCr56HWN

