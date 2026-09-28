import torch
import torch.nn as nn
from darcy.config import *  

class AFPINN(nn.Module):
    def __init__(self, kx, ky, phi, coeff, bias):
        super().__init__()
        K = len(kx)
        self.kx    = nn.Parameter(kx.reshape(K,1))
        self.ky    = nn.Parameter(ky.reshape(K,1))
        self.phi   = nn.Parameter(phi.reshape(K,1))
        self.coeff = nn.Parameter(coeff.reshape(1,K))
        self.bias  = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, x, y):          
        x = x.view(-1,1);  y = y.view(-1,1)
        theta   = x @ self.kx.t() + y @ self.ky.t() + self.phi.t()   # (N,K)
        cos_th  = torch.cos(theta)
        sin_th  = torch.sin(theta)
        output  = (cos_th @ self.coeff.t()).squeeze() + self.bias
        dudx   = -(self.kx.t() * sin_th    @ self.coeff.t()).squeeze()
        du_dy   = -(self.ky.t() * sin_th    @ self.coeff.t()).squeeze()
        d2u_dx2 = -(self.kx.t()**2 * cos_th @ self.coeff.t()).squeeze()
        d2u_dy2 = -(self.ky.t()**2 * cos_th @ self.coeff.t()).squeeze()
        return output, dudx, du_dy, d2u_dx2, d2u_dy2


