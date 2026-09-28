import torch
import torch.nn as nn
from poisson.config import *  # expects device, seed

class AWPINN(nn.Module):
    def __init__(self, wx, bx, wy, by, coeff, bias):
        super(AWPINN, self).__init__()
        num_wavelets = len(wx)
        
        # Make these parameters trainable
        self.wx = nn.Parameter(wx.reshape(num_wavelets, 1))
        self.bx = nn.Parameter(-bx)
        self.wy = nn.Parameter(wy.reshape(num_wavelets, 1))
        self.by = nn.Parameter(-by)
    
        self.output_weight = nn.Parameter(coeff.reshape(1, -1))
        self.output_bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x, y):

        # with torch.no_grad():
        x = x.view(-1, 1)
        y = y.view(-1, 1)
            
        x_transformed = x @ self.wx.t() + self.bx
        y_transformed = y @ self.wy.t() + self.by
        
        x_exp = torch.exp(-x_transformed**2 / 2)
        y_exp = torch.exp(-y_transformed**2 / 2)
        
        x_wavelets = -x_transformed * x_exp
        y_wavelets = -y_transformed * y_exp
        
        d2x_wavelets = (self.wx.t()**2) * x_transformed * (3 - x_transformed**2) * x_exp
        d2y_wavelets = (self.wy.t()**2) * y_transformed * (3 - y_transformed**2) * y_exp
        scale = torch.sqrt(torch.clamp(self.wx.t() * self.wy.t(), min=1e-8))
        
        wavelets = scale * x_wavelets * y_wavelets
        output = (wavelets @ self.output_weight.t()).squeeze() + self.output_bias
        
        # Derivatives
        d2u_dx2 = scale * (d2x_wavelets * y_wavelets) @ self.output_weight.t()
        d2u_dy2 = scale * (x_wavelets * d2y_wavelets) @ self.output_weight.t()

        
        return output, d2u_dx2.squeeze(), d2u_dy2.squeeze()