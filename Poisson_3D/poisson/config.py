import torch
import torch.nn as nn

global device
device = torch.device('cuda:1')
torch.manual_seed(121)

class DataConfig:
    def __init__(self):

        self.n_collocation = 20000
        self.n_validation = 128
        self.n_boundary = 2000
        self.n_test = 64
        self.n_fno = 64
        
        self.x_lower = 0
        self.x_upper = 1
        self.y_lower = 0
        self.y_upper = 1
        self.z_lower = 0
        self.z_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_collocation = (torch.rand(self.n_collocation) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        z_collocation = (torch.rand(self.n_collocation) * (self.z_upper - self.z_lower) + self.z_lower).to(self.device)

        
        # Boundary condition points
        z_bc = (torch.rand(self.n_boundary) * (self.z_upper - self.z_lower) + self.z_lower).to(self.device)
        x_bc = (torch.rand(self.n_boundary) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_bc = (torch.rand(self.n_boundary) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)
        y_bc_bottom = self.y_lower * torch.ones(self.n_boundary).to(self.device)
        y_bc_top = self.y_upper * torch.ones(self.n_boundary).to(self.device)
        z_bc_back = self.z_lower * torch.ones(self.n_boundary).to(self.device)
        z_bc_front = self.z_upper * torch.ones(self.n_boundary).to(self.device)


        # Validation points
        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_validation = (torch.rand(self.n_validation) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        z_validation = (torch.rand(self.n_validation) * (self.z_upper - self.z_lower) + self.z_lower).to(self.device)

        #test points 64*64*64
        x_test = torch.linspace(self.x_lower, self.x_upper, self.n_test).to(self.device)
        y_test = torch.linspace(self.y_lower, self.y_upper, self.n_test).to(self.device)
        z_test = torch.linspace(self.z_lower, self.z_upper, self.n_test).to(self.device)
        x_test, y_test, z_test = torch.meshgrid(x_test, y_test, z_test)

        x_fno = torch.linspace(self.x_lower, self.x_upper, self.n_fno).to(self.device)
        y_fno = torch.linspace(self.y_lower, self.y_upper, self.n_fno).to(self.device)
        z_fno = torch.linspace(self.z_lower, self.z_upper, self.n_fno).to(self.device)
        x_fno, y_fno, z_fno = torch.meshgrid(x_fno, y_fno, z_fno)
        x_fno = x_fno.reshape(-1)
        y_fno = y_fno.reshape(-1)
        z_fno = z_fno.reshape(-1)


        
        return {
            'domain': (self.x_lower, self.x_upper, self.y_lower, self.y_upper, self.z_lower, self.z_upper),  
            'collocation': (self.n_collocation, x_collocation, y_collocation, z_collocation),
            'boundary': (x_bc_left, x_bc_right, x_bc, y_bc_bottom, y_bc_top, y_bc, z_bc_back, z_bc_front, z_bc),
            'validation': (x_validation, y_validation, z_validation),
            'test': (self.n_test, x_test, y_test, z_test),
            'fno': (self.n_fno, x_fno, y_fno, z_fno)
        }
    

config = DataConfig()
points = config.generate_training_points()

# Access the points as needed
x_lower, x_upper, y_lower, y_upper, z_lower, z_upper = points['domain']
n_collocation, x_collocation, y_collocation, z_collocation = points['collocation']
x_bc_left, x_bc_right, x_bc, y_bc_bottom, y_bc_top, y_bc, z_bc_back, z_bc_front, z_bc = points['boundary']
x_validation, y_validation, z_validation = points['validation']
n_test, x_test, y_test, z_test = points['test']
n_fno, x_fno, y_fno, z_fno = points['fno']