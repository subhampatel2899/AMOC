import torch
import numpy as np
from darcy.config import *

# ===================== WAVELET FAMILY =====================
def wavelet_family(Jx, Jy, a):
    family = torch.tensor([
        (2**jx, 2**jy, kx, ky)
        for jx in Jx for jy in Jy
        for kx in range(int(torch.floor((x_lower-a)*2**(jx))),
                        int(torch.ceil((x_upper+a)*2**(jx))) + 1)
        for ky in range(int(torch.floor((y_lower-a)*2**(jy))),
                        int(torch.ceil((y_upper+a)*2**(jy))) + 1)
    ], device=device)

    return family

# ===================== GAUSSIAN WAVELET =====================
def gaussian(x, y, jx, jy, kx, ky):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]

    ex = torch.exp(-(X**2 + Y**2)/2)

    # norm = 2 / torch.sqrt(torch.tensor(torch.pi, device=x.device))
    scale = torch.sqrt(jx[:, None] * jy[:, None])

    return scale * X * Y * ex