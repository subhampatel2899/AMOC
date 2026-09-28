import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import default_timer
from darcy.config import *  
from darcy.parameter import *
from darcy.waveletfamily import *
from darcy.coeff_selection import *
from Darcy.darcy.wamoc import AWPINN
from tqdm import tqdm
import numpy as np
seed = 121
torch.manual_seed(seed)
np.random.seed(seed)
# load data 
data_test = np.load("data/test_dataset_out.npz", allow_pickle=True)
k = torch.tensor(data_test["k"], dtype=torch.float32)
f = torch.tensor(data_test["f"], dtype=torch.float32)
p_exact = torch.tensor(data_test["p"], dtype=torch.float32)
perm_params = data_test["perm_params"]
forcing_params = data_test["forcing_params"]

dataset = np.load("data/darcy_pred_fno_out.npz", allow_pickle=True)
p_fno_pred = torch.tensor(dataset["predictions"], device=device)
dx = (x_fno[-1] - x_fno[0])/(n_fno-1)
dy = (y_fno[-1] - y_fno[0])/(n_fno-1)
Jx = torch.arange(0, 6)
Jy = torch.arange(0, 6)
a = 0.3
family = wavelet_family(Jx, Jy, a)
jx, jy, kx, ky = family[:,0], family[:,1], family[:,2], family[:,3]

Err =[]
def train_awpinn_for_fno(i):
    chunk_size = 128
    p_vec = p_fno_pred[i].reshape(-1)
    b = apply_WT(p_vec, x_fno.to(device), y_fno.to(device), jx, jy, kx, ky, chunk_size=chunk_size) * dx * dy
    coeff = conjugate_gradient(b, dx, dy, x_fno.to(device), y_fno.to(device), jx, jy, kx, ky, max_iter=1, tol=1e-2)
    u_recon_vec = apply_W(coeff, x_fno.to(device), y_fno.to(device), jx, jy, kx, ky, chunk_size=chunk_size)
    bias = (u_recon_vec - p_vec).mean()
    top_indices = torch.nonzero(abs(coeff) > torch.quantile(abs(coeff), 0.7), as_tuple=True)[0]

    selected_indices = torch.unique(top_indices)

    coeff_new = coeff[selected_indices]
    family_new = family[selected_indices]

    print("coeff_len: ", len(coeff_new))
    print("Family len: ", len(family_new))

    wx = family_new[:,0].float().to(device)
    bx = family_new[:,2].float().to(device)
    wy = family_new[:,1].float().to(device)
    by = family_new[:,3].float().to(device)

    AWPINN_model = AWPINN(wx, bx, wy, by, coeff_new, bias).to(device)

    optimizer_AWPINN = torch.optim.LBFGS(AWPINN_model.parameters(), 
                                lr=1.0,                            
                                max_iter=5000,
                                max_eval=10**10,
                                tolerance_grad=1e-10,
                                tolerance_change=1e-10,
                                history_size=50,
                                line_search_fn=None)
    x_interior = x_collocation.clone()
    y_interior = y_collocation.clone()
    k_p, dk_dx_p, dk_dy_p = permeability_derivatives(x_interior, y_interior, perm_params[i], device=device)
    rhs = forcing_function(x_interior, y_interior, forcing_params[i])
    u_bc_left = torch.zeros_like(y_bc)
    u_bc_right = torch.zeros_like(y_bc)
    u_bc_bottom = torch.zeros_like(x_bc)
    u_bc_top = torch.zeros_like(x_bc)
    exact = p_exact[i] 
    def awpinn_loss():   

        u, u_x, u_y, u_xx, u_yy = AWPINN_model(x_interior, y_interior)

        pde_loss = (torch.mean((- (k_p * (u_yy + u_xx) + dk_dx_p * u_x + dk_dy_p * u_y) - rhs)**2))
        
        u_pred_bc_left, _, _, _, _ = AWPINN_model(x_bc_left, y_bc)
        u_pred_bc_right, _, _, _, _ = AWPINN_model(x_bc_right, y_bc)
        u_pred_bc_bottom, _, _, _, _ = AWPINN_model(x_bc, y_bc_bottom)
        u_pred_bc_top, _, _, _, _ = AWPINN_model(x_bc, y_bc_top)

        
        bc_loss = torch.mean((u_pred_bc_left - u_bc_left) ** 2) +\
                torch.mean((u_pred_bc_right - u_bc_right) ** 2) +\
                torch.mean((u_pred_bc_bottom - u_bc_bottom) ** 2) + \
                torch.mean((u_pred_bc_top - u_bc_top) ** 2)
        lam_bc = 500 
        total_loss = pde_loss + lam_bc * bc_loss
        return total_loss, pde_loss, bc_loss

    def train_awpinn(optimizer):

        global itr
        itr = 0

        def closure():
            global itr
            optimizer.zero_grad()
            total_loss, pde_loss, bc_loss = awpinn_loss()

            total_loss.backward()

            if itr % 1000 == 0 or itr == 100:
                with torch.no_grad():

                    numerical, _, _, _, _ = AWPINN_model(x_test.to(device), y_test.to(device))
                
                    errL2 = (torch.sum(torch.abs(exact.reshape(-1).to(device)-numerical)**2))**0.5 / (torch.sum(torch.abs(exact.reshape(-1).to(device))**2))**0.5
                    errMax = torch.max(torch.abs(exact.reshape(-1).to(device)-numerical))

                print(f'Epoch[{itr}]  '
                        f'Total Loss: {total_loss.item():.6f}, '
                        f'PDE Loss: {pde_loss.item():.6f}, '
                        f'BC Loss: {bc_loss.item():.6f}\n\t\t'
                        f'RelativeL2: {errL2},\t'
                        f'Max: {errMax}\n' )
                
                torch.cuda.empty_cache()

                
            itr += 1

            return total_loss

        loss = optimizer.step(closure)

    loss = train_awpinn(optimizer_AWPINN)
    with torch.no_grad():
        numerical, _, _, _, _ = AWPINN_model(x_test.to(device), y_test.to(device))
    return numerical.cpu(), exact.cpu()

i = 0 #instance index
numerical, exact_test = train_awpinn_for_fno(i)
ErrL2 = (torch.sum(torch.abs(exact_test.reshape(-1)-numerical.reshape(-1))**2))**0.5 / (torch.sum(torch.abs(exact_test.reshape(-1))**2))**0.5
ErrMax = torch.max(torch.abs(exact_test.reshape(-1)-numerical.reshape(-1)))