import torch
import torch.nn as nn
from poisson.config import *

# ===================== WAVELET FAMILY =====================
def wavelet_family(Jx, Jy, Jt, a):
    family = torch.tensor([
        (2**jx, 2**jy, 2**jt, kx, ky, kt)
        for jx in Jx for jy in Jy for jt in Jt 
        for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1)
        for ky in range(int(torch.floor((y_lower-a)*2**(jy))), int(torch.ceil((y_upper+a)*2**(jy))) + 1) 
        for kt in range(int(torch.floor((z_lower-a)*2**(jt))), int(torch.ceil((z_upper+a)*2**(jt))) + 1)
    ], device=device)

    return len(family), family


def gaussian(x, y, z, jx, jy, jt, kx, ky, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    T = jt[:, None] * z[None, :] - kt[:, None]

    ex = torch.exp(-(X**2 + Y**2 + T**2) / 2)
    norm = torch.sqrt(jx[:, None] * jy[:, None] * jt[:, None])

    return norm * (-X * Y * T * ex)  # (chunk, N)



