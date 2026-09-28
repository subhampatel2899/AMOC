import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


global device
device = torch.device('cuda:0')


class DataConfig:
    def __init__(self):

        self.n_collocation = 50000
        self.n_validation = 128
        self.n_boundary = 4096   # must be perfect square (e.g., 64x64)
        self.n_test = 64
        self.n_fno = 64
        
        self.x_lower = -1
        self.x_upper = 1
        self.y_lower = -1
        self.y_upper = 1
        self.z_lower = -1
        self.z_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""

        # ------------------ Collocation ------------------
        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_collocation = (torch.rand(self.n_collocation) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        z_collocation = (torch.rand(self.n_collocation) * (self.z_upper - self.z_lower) + self.z_lower).to(self.device)

        # ------------------ Boundary (UNIFORM GRID) ------------------
        n_side = int(self.n_boundary ** 0.5)

        x_lin = torch.linspace(self.x_lower, self.x_upper, n_side).to(self.device)
        y_lin = torch.linspace(self.y_lower, self.y_upper, n_side).to(self.device)
        z_lin = torch.linspace(self.z_lower, self.z_upper, n_side).to(self.device)

        # X boundaries
        y_grid, z_grid = torch.meshgrid(y_lin, z_lin, indexing='ij')

        x_bc_left  = self.x_lower * torch.ones_like(y_grid).to(self.device)
        x_bc_right = self.x_upper * torch.ones_like(y_grid).to(self.device)

        x_bc_left  = x_bc_left.reshape(-1)
        x_bc_right = x_bc_right.reshape(-1)
        y_bc       = y_grid.reshape(-1)
        z_bc       = z_grid.reshape(-1)

        # Y boundaries
        x_grid, z_grid = torch.meshgrid(x_lin, z_lin, indexing='ij')

        y_bc_bottom = self.y_lower * torch.ones_like(x_grid).to(self.device)
        y_bc_top    = self.y_upper * torch.ones_like(x_grid).to(self.device)

        y_bc_bottom = y_bc_bottom.reshape(-1)
        y_bc_top    = y_bc_top.reshape(-1)
        x_bc        = x_grid.reshape(-1)
        z_bc_y      = z_grid.reshape(-1)

        # Z boundaries
        x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing='ij')

        z_bc_back  = self.z_lower * torch.ones_like(x_grid).to(self.device)
        z_bc_front = self.z_upper * torch.ones_like(x_grid).to(self.device)

        z_bc_back  = z_bc_back.reshape(-1)
        z_bc_front = z_bc_front.reshape(-1)
        x_bc_z     = x_grid.reshape(-1)
        y_bc_z     = y_grid.reshape(-1)

        # ------------------ Validation ------------------
        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_validation = (torch.rand(self.n_validation) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        z_validation = (torch.rand(self.n_validation) * (self.z_upper - self.z_lower) + self.z_lower).to(self.device)

        # ------------------ Test ------------------
        x_test = torch.linspace(self.x_lower, self.x_upper, self.n_test).to(self.device)
        y_test = torch.linspace(self.y_lower, self.y_upper, self.n_test).to(self.device)
        z_test = torch.linspace(self.z_lower, self.z_upper, self.n_test).to(self.device)
        x_test, y_test, z_test = torch.meshgrid(x_test, y_test, z_test, indexing='ij')

        # ------------------ FNO grid ------------------
        x_fno = torch.linspace(self.x_lower, self.x_upper, self.n_fno).to(self.device)
        y_fno = torch.linspace(self.y_lower, self.y_upper, self.n_fno).to(self.device)
        z_fno = torch.linspace(self.z_lower, self.z_upper, self.n_fno).to(self.device)
        x_fno, y_fno, z_fno = torch.meshgrid(x_fno, y_fno, z_fno, indexing='ij')

        x_fno = x_fno.reshape(-1)
        y_fno = y_fno.reshape(-1)
        z_fno = z_fno.reshape(-1)

        return {
            'domain': (self.x_lower, self.x_upper, self.y_lower, self.y_upper, self.z_lower, self.z_upper),

            'collocation': (self.n_collocation, x_collocation, y_collocation, z_collocation),

            # same style as yours (flat tensors)
            'boundary': (
                x_bc_left, x_bc_right, y_bc,          # x boundaries
                y_bc_bottom, y_bc_top, x_bc,          # y boundaries
                z_bc_back, z_bc_front, z_bc           # z boundaries
            ),

            'validation': (x_validation, y_validation, z_validation),

            'test': (self.n_test, x_test, y_test, z_test),

            'fno': (self.n_fno, x_fno, y_fno, z_fno)
        }


# ------------------ Usage ------------------
config = DataConfig()
points = config.generate_training_points()

x_lower, x_upper, y_lower, y_upper, z_lower, z_upper = points['domain']

n_collocation, x_collocation, y_collocation, z_collocation = points['collocation']

x_bc_left, x_bc_right, y_bc, y_bc_bottom, y_bc_top, x_bc, z_bc_back, z_bc_front, z_bc = points['boundary']

x_validation, y_validation, z_validation = points['validation']

n_test, x_test, y_test, z_test = points['test']

n_fno, x_fno, y_fno, z_fno = points['fno']