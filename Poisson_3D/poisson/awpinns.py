import torch
import torch.nn as nn
from poisson.config import *

class AWPINN(nn.Module):
    def __init__(self, wx, bx, wy, by, wz, bz, coeff, bias):
        super(AWPINN, self).__init__()
        num_wavelets = len(wx)
        
        
        self.register_buffer("wx", wx.reshape(num_wavelets, 1).contiguous())
        self.register_buffer("bx", -bx.reshape(1, -1).contiguous())
        self.register_buffer("wy", wy.reshape(num_wavelets, 1).contiguous())
        self.register_buffer("by", -by.reshape(1, -1).contiguous())
        self.register_buffer("wz", wz.reshape(num_wavelets, 1).contiguous())
        self.register_buffer("bz", -bz.reshape(1, -1).contiguous())
        
        self.output_weight = nn.Parameter(coeff.reshape(1, -1))
        self.output_bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x, y, z, compute_derivatives=True):

        # with torch.no_grad():
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        z = z.view(-1, 1)
        
        x_transformed = x @ self.wx.t() + self.bx
        y_transformed = y @ self.wy.t() + self.by
        z_transformed = z @ self.wz.t() + self.bz  

        x_exp = torch.exp(-x_transformed**2 / 2)
        y_exp = torch.exp(-y_transformed**2 / 2)
        z_exp = torch.exp(-z_transformed**2 / 2)
        
        x_wavelets = -x_transformed * x_exp
        y_wavelets = -y_transformed * y_exp
        z_wavelets = -z_transformed * z_exp

        
        scale = torch.sqrt(torch.clamp(self.wx.t() * self.wy.t() * self.wz.t(), min=1e-12))  # (1, num_wavelets)
        # scale = torch.sqrt(self.wx.t() * self.wy.t() * self.wz.t())  # (1, num_wavelets)
        wavelets = scale * x_wavelets * y_wavelets * z_wavelets
        output = (wavelets @ self.output_weight.t()).squeeze() + self.output_bias

        if not compute_derivatives:
            return output, None, None, None
        
        # Derivatives
        d2x_wavelets = (self.wx.t()**2) * x_transformed * (3 - x_transformed**2) * x_exp
        d2u_dx2 = scale * (d2x_wavelets * y_wavelets * z_wavelets) @ self.output_weight.t()
        del d2x_wavelets

        d2y_wavelets = (self.wy.t()**2) * y_transformed * (3 - y_transformed**2) * y_exp
        d2u_dy2 = scale * (x_wavelets * d2y_wavelets * z_wavelets) @ self.output_weight.t()
        del d2y_wavelets


        d2z_wavelets = (self.wz.t()**2) * z_transformed * (3 - z_transformed**2) * z_exp
        d2u_dz2 = scale * (x_wavelets * y_wavelets * d2z_wavelets) @ self.output_weight.t()
        del d2z_wavelets, scale, x_wavelets, y_wavelets, z_wavelets, x_exp, y_exp, z_exp
        del x_transformed, y_transformed, z_transformed

        return output, d2u_dx2.squeeze(), d2u_dy2.squeeze(), d2u_dz2.squeeze()