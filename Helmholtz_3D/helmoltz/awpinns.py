import torch
import torch.nn as nn




class AWPINN(nn.Module):
    def __init__(self, wx, bx, wy, by, wz, bz, coeff, bias):
        super(AWPINN, self).__init__()
        num_wavelets = len(wx)
        
        self.wx = nn.Parameter(wx.reshape(num_wavelets, 1))
        self.bx = nn.Parameter(-bx)
        self.wy = nn.Parameter(wy.reshape(num_wavelets, 1))
        self.by = nn.Parameter(-by)
        self.wz = nn.Parameter(wz.reshape(num_wavelets, 1))
        self.bz = nn.Parameter(-bz)
        self.output_weight = nn.Parameter(coeff.reshape(1, -1))
        self.output_bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x, y, z):

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

        

        scale = torch.sqrt(self.wx.t() * self.wy.t() * self.wz.t())  # (1, num_wavelets)
        wavelets = scale * x_wavelets * y_wavelets * z_wavelets
        output = (wavelets @ self.output_weight.t()).squeeze() + self.output_bias
        
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

        return output, d2u_dx2.squeeze(), d2u_dy2.squeeze(), d2u_dz2.squeeze()