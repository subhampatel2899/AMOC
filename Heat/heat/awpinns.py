import torch
import torch.nn as nn
from heat.config import *  # expects device, seed
class AWPINN(nn.Module):
    def __init__(self, wx, bx, wt, bt, coeff, bias):
        super(AWPINN, self).__init__()
        num_wavelets = len(wx)
        
        # Make these parameters trainable
        self.wx = nn.Parameter(wx.reshape(num_wavelets, 1))
        self.bx = nn.Parameter(-bx)
        self.wt = nn.Parameter(wt.reshape(num_wavelets, 1))
        self.bt = nn.Parameter(-bt)
        
        self.output_weight = nn.Parameter(coeff.reshape(1, -1))
        self.output_bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x, t):

        # with torch.no_grad():
        x = x.view(-1, 1)
        t = t.view(-1, 1)
            
        x_transformed = x @ self.wx.t() + self.bx
        t_transformed = t @ self.wt.t() + self.bt
        
        x_exp = torch.exp(-x_transformed**2 / 2)
        t_exp = torch.exp(-t_transformed**2 / 2)
        
        x_wavelets = -x_transformed * x_exp
        t_wavelets = -t_transformed * t_exp
        
        dt_wavelets = self.wt.t() * (t_transformed**2 - 1) * t_exp
        d2x_wavelets = (self.wx.t()**2) * x_transformed * (3 - x_transformed**2) * x_exp
        scale = torch.sqrt(torch.clamp(self.wx.t() * self.wt.t(), min=1e-10))
        
        # scale = torch.sqrt(self.wx.t() * self.wt.t())
        wavelets = scale * x_wavelets * t_wavelets
        output = (wavelets @ self.output_weight.t()).squeeze() + self.output_bias
        
        # Derivatives
        du_dt = scale * (x_wavelets * dt_wavelets) @ self.output_weight.t()
        d2u_dx2 = scale * (d2x_wavelets * t_wavelets) @ self.output_weight.t()

        
        return output, du_dt.squeeze(), d2u_dx2.squeeze()