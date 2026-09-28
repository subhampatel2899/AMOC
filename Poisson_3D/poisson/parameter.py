import numpy as np
import torch
from poisson.config import *

def analytical(x, y, z, params):
    u = 0.0
    for p in params:
        r2 = (x - p['x0'])**2 + (y - p['y0'])**2 + (z - p['z0'])**2
        u += p['A'] * torch.exp(-p['alpha'] * r2)
    return u

def right_side(x, y, z, params):
    f = 0.0
    for p in params:
        A, alpha = p['A'], p['alpha']
        r2 = (x - p['x0'])**2 + (y - p['y0'])**2 + (z - p['z0'])**2
        g      = A * torch.exp(-alpha * r2)
        lap_g  = g * (4 * alpha**2 * r2 - 6 * alpha)   
        f     += -lap_g                                 
    return f