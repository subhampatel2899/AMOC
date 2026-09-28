import torch
import torch.nn as nn
from poisson.config import *  

class AFPINN(nn.Module):
    def __init__(self, kx, ky, kz, phi, coeff, bias):
        super().__init__()
        K = len(kx)
        self.kx    = nn.Parameter(kx.reshape(K, 1))
        self.ky    = nn.Parameter(ky.reshape(K, 1))
        self.kz    = nn.Parameter(kz.reshape(K, 1))
        self.phi   = nn.Parameter(phi.reshape(K, 1))
        self.coeff = nn.Parameter(coeff.reshape(1, K))
        self.bias  = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, x, y, z):
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        z = z.view(-1, 1)
        # (N, K) phase
        theta  = x @ self.kx.t() + y @ self.ky.t() + z @ self.kz.t() + self.phi.t()
        cos_th = torch.cos(theta)

        output  = (cos_th @ self.coeff.t()).squeeze() + self.bias
        u_xx    = -(self.kx.t()**2 * cos_th @ self.coeff.t()).squeeze()
        u_yy    = -(self.ky.t()**2 * cos_th @ self.coeff.t()).squeeze()
        u_zz    = -(self.kz.t()**2 * cos_th @ self.coeff.t()).squeeze()
        return output, u_xx, u_yy, u_zz