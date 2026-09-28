import torch
import torch.nn as nn
from heat.config import *  

class AFPINN(nn.Module):
    def __init__(self, kx, kt, phi, coeff, bias):
        super().__init__()
        K = len(kx)
        self.kx    = nn.Parameter(kx.reshape(K,1))
        self.kt    = nn.Parameter(kt.reshape(K,1))
        self.phi   = nn.Parameter(phi.reshape(K,1))
        self.coeff = nn.Parameter(coeff.reshape(1,K))
        self.bias  = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, x, t):          
        x = x.view(-1,1);  t = t.view(-1,1)
        theta   = x @ self.kx.t() + t @ self.kt.t() + self.phi.t()   # (N,K)
        cos_th  = torch.cos(theta)
        sin_th  = torch.sin(theta)
        output  = (cos_th @ self.coeff.t()).squeeze() + self.bias
        du_dt   = -(self.kt.t() * sin_th    @ self.coeff.t()).squeeze()
        d2u_dx2 = -(self.kx.t()**2 * cos_th @ self.coeff.t()).squeeze()
        return output, du_dt, d2u_dx2


