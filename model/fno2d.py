import torch
import torch.nn as nn
from model.spectralconv2d import SpectralConv2d
import numpy as np
import torch.nn.functional as F

class FNO2d(nn.Module):
    def __init__(self, width, modes, layers, size, in_channel, grid_range):
        super(FNO2d, self).__init__()
        self.width = width
        self.modes = modes
        self.layers = layers
        self.size = size
        self.in_channel = in_channel
        self.grid_range = grid_range

        self.fc0 = nn.Linear(self.in_channel, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.fno_conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        for _ in range(self.layers):
            self.fno_conv.append(SpectralConv2d(self.width, self.width, self.modes, self.modes))
            self.w.append(nn.Conv2d(self.width, self.width, 1))

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for index, (convl1, wl) in enumerate(zip(self.fno_conv, self.w)):
            fno_out = convl1(x)
            simple_conv = wl(x)
            x = fno_out +  simple_conv
            if index != self.layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x


    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
