# Adaptive Multiscale Operator Correction (AMOC)

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-8C1D40.svg)](https://neurips.cc/virtual/2026/loc/sydney/poster/149265)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-b31b1b.svg)](https://openreview.net/forum?id=rbb6tqGnFf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyTorch implementation of **Adaptive Multiscale Operator Correction (AMOC)** for PDE solution correction using a neural-operator prediction followed by physics-informed refinement in a sparse spectral representation.

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

## Data

Benchmark datasets and pre-generated files are available on Google Drive:
- [Download Datasets (Google Drive)](https://drive.google.com/drive/folders/1itTmOVgtIAs53xGvSjgPpoQ9zCr56HWN)

## Citation

If you find this code or our paper useful in your research, please cite:

```bibtex
@inproceedings{patel2026adaptive,
  title     = {Adaptive Multiscale Operator Correction via Learned Spectral Subspace and Physics-Informed Optimization},
  author    = {Subham Patel and Himanshu Pandey and Ratikanta Behera},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=rbb6tqGnFf}
}
```

## Acknowledgements & Related Work

This implementation builds on ideas and software ecosystems from prior work on neural operators and wavelet-based physics-informed neural networks:

```bibtex
@inproceedings{li2021_fno,
  title     = {Fourier Neural Operator for Parametric Partial Differential Equations},
  author    = {Zongyi Li and Nikola Borislavov Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://openreview.net/forum?id=c8P9NQVtmnO}
}

@article{PANDEY2026108860,
  title   = {An efficient wavelet-based physics-informed neural network for multiscale problems},
  author  = {Himanshu Pandey and Anshima Singh and Ratikanta Behera},
  journal = {Neural Networks},
  volume  = {200},
  pages   = {108860},
  year    = {2026},
  issn    = {0893-6080},
  doi     = {10.1016/j.neunet.2026.108860},
  url     = {https://www.sciencedirect.com/science/article/pii/S0893608026003217}
}
```

## License

This project is released under the [MIT License](LICENSE).
