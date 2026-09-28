import torch
import numpy as np
from darcy.config import *

def permeability_derivatives(xx, yy, params, device=device):

    k = torch.ones_like(xx)

    dk_dx = torch.zeros_like(xx)
    dk_dy = torch.zeros_like(xx)

    for g in params:
        x0 = g["x0"]
        y0 = g["y0"]
        sigma = g["sigma"]
        amp = g["amp"]

        r2 = (xx-x0)**2 + (yy-y0)**2

        exp_term = torch.exp(-r2/(2*sigma**2))

        k += amp*exp_term

        dk_dx += amp*exp_term*(-(xx-x0)/(sigma**2))
        dk_dy += amp*exp_term*(-(yy-y0)/(sigma**2))

    return k, dk_dx, dk_dy


def forcing_function(xx, yy, forcing_params, device=device):
    f = torch.zeros_like(xx, device=device)

    for param in forcing_params:

        nx = param["nx"]
        ny = param["ny"]
        amp = param["amp"]

        f += amp * torch.sin(nx * torch.pi * xx) * torch.sin(ny * torch.pi * yy)

    return f