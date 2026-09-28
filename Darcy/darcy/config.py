import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from scipy.stats import qmc

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
global device
device = torch.device('cuda:0')
# Global device configuration
torch.manual_seed(121)

class DataConfig:
    def __init__(self):
        # Sample sizes
        self.n_collocation = 10000
        self.n_validation = 1000
        self.n_boundary = 3000
        self.n_test = 128
        self.n_fno = 64
        
        # Domain bounds
        self.x_lower = 0
        self.x_upper = 1
        self.y_lower = 0
        self.y_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        # Collocation points

        
        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_collocation = (torch.rand(self.n_collocation) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        
        # Initial condition points
        x_bc = (torch.rand(self.n_boundary) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_bc_bottom = self.y_lower * torch.ones(self.n_boundary).to(self.device)
        y_bc_top = self.y_upper * torch.ones(self.n_boundary).to(self.device)
        
        # Boundary condition points
        y_bc = (torch.rand(self.n_boundary) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)


        # Validation points
        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower)
        y_validation = (torch.rand(self.n_validation) * (self.y_upper - self.y_lower) + self.y_lower)

        # Testing and Plotting points
        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        ytest = torch.linspace(self.y_lower, self.y_upper, self.n_test)
            
        x_grid, y_grid = torch.meshgrid(xtest, ytest)
        x_test = x_grid.reshape(-1)
        y_test = y_grid.reshape(-1)

        xfno = torch.linspace(self.x_lower, self.x_upper, self.n_fno)
        yfno = torch.linspace(self.y_lower, self.y_upper, self.n_fno)
        x_fno_grid, y_fno_grid = torch.meshgrid(xfno, yfno)
        x_fno = x_fno_grid.reshape(-1).to(self.device)
        y_fno = y_fno_grid.reshape(-1).to(self.device)
        
        return {
            'domain': (self.x_lower, self.x_upper, self.y_lower, self.y_upper),  
            'collocation': (self.n_collocation, x_collocation, y_collocation),
            'validation': (x_validation, y_validation),
            'boundary': (y_bc, x_bc_left, x_bc_right, x_bc, y_bc_bottom, y_bc_top),
            'test': (self.n_test, x_test, y_test),
            'fno': (self.n_fno, x_fno, y_fno)
        }
    

config = DataConfig()
points = config.generate_training_points()

# Access the points as needed
x_lower, x_upper, y_lower, y_upper = points['domain']
n_collocation, x_collocation, y_collocation = points['collocation']
x_validation, y_validation = points['validation']
y_bc, x_bc_left, x_bc_right, x_bc, y_bc_bottom, y_bc_top = points['boundary']
n_test, x_test, y_test = points['test']
n_fno, x_fno, y_fno = points['fno']