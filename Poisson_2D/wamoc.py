import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
# matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
from timeit import default_timer
from poisson.config import *  # expects device, seed
from poisson.waveletfamily import *
from poisson.coeff_selection import *
from poisson.awpinns import AWPINN
from tqdm import tqdm

seed = 121
torch.manual_seed(seed)
np.random.seed(seed)
# Data Loader
test_data = np.load("data/poisson_data_in.npz", allow_pickle=True)
eps_test = torch.tensor(test_data["eps"], dtype=torch.float32)
u_test = torch.tensor(test_data["u"], dtype=torch.float32)
pred_data = np.load("data/poisson_pred_fno_in.npz")
u_fno_pred = torch.tensor(pred_data["pred"], dtype=torch.float32)
dx = (x_fno[-1] - x_fno[0])/(n_fno-1)
dy = (y_fno[-1] - y_fno[0])/(n_fno-1)
Jx = torch.arange(0, 7)
Jy = torch.arange(0, 6)
a = 0.2
family = wavelet_family(Jx, Jy, a, x_lower, x_upper, y_lower, y_upper)
jx, jy, kx, ky = family[:,0], family[:,1], family[:,2], family[:,3]
chunk_size = 256
Err = []
def train_awpinn_for_fno(i):
    u_vec = u_fno_pred[i].reshape(-1).to(device)

    b = apply_WT(u_vec, x_fno.to(device), y_fno.to(device), jx, jy, kx, ky, chunk_size=chunk_size) * dx * dy
    coeff = conjugate_gradient(b, dx, dy, x_fno.to(device), y_fno.to(device), jx, jy, kx, ky, max_iter=10, tol=1e-2)

    u_recon_vec = apply_W(coeff, x_fno.to(device), y_fno.to(device), jx, jy, kx, ky, chunk_size=chunk_size)
    u_recon = u_recon_vec.reshape(u_fno_pred.shape[-2], u_fno_pred.shape[-1])
    bias = (u_recon_vec - u_vec).mean()
    print("Bias: ", bias.item())
    top_indices = torch.nonzero(abs(coeff) > torch.quantile(abs(coeff), 0.9), as_tuple=True)[0]

    selected_indices = torch.unique(top_indices)

    coeff_new = coeff[selected_indices]
    family_new = family[selected_indices]

    print("coeff_len: ", len(coeff_new))
    print("Family len: ", len(family_new))

    wx = family_new[:,0].float().to(device)
    bx = family_new[:,2].float().to(device)
    wy = family_new[:,1].float().to(device)
    by = family_new[:,3].float().to(device)
    coeff_new = coeff_new.detach()
    wx = wx.detach()
    bx = bx.detach()
    wy = wy.detach()
    by = by.detach()
    bias = bias.detach()
    AWPINN_model = AWPINN(wx, bx, wy, by, coeff_new, bias).to(device)

    optimizer_AWPINN = torch.optim.LBFGS(AWPINN_model.parameters(), 
                            lr=1.0,                            
                            max_iter=5*10**3,
                            max_eval=10**10,
                            tolerance_grad=1e-10,
                            tolerance_change=1e-10,
                            history_size=50,
                            line_search_fn=None)
    
    
    x_interior = x_collocation.clone()
    y_interior = y_collocation.clone()
    e = eps_test[i,0,0].cpu().item()
    print(e)
    def analytical(x,y):
        x_t = (x-0.5)**2
        y_t = 1000 + y**2
        exp = torch.exp(- x_t / (2 * e**2))

        return 1 + y_t * exp


    def right_side(x,y):
        x_t = (x-0.5)**2
        y_t = 1000 + y**2
        exp = torch.exp(- x_t / (2 * e**2))

        return 2*exp + y_t * (x_t*exp/e**4 - exp/e**2)



    u_bc_left = analytical(x_bc_left, y_bc)
    u_bc_right = analytical(x_bc_right, y_bc)
    u_bc_bottom = analytical(x_bc, y_bc_bottom)
    u_bc_top = analytical(x_bc, y_bc_top)

    rhs = right_side(x_collocation, y_collocation)
    exact_validation = analytical(x_validation, y_validation)
    exact_test = analytical(x_test, y_test).reshape(n_test, n_test)


    def awpinn_loss():   

        u, u_xx, u_yy = AWPINN_model(x_interior, y_interior)

        pde_loss = (torch.mean((u_yy + u_xx - rhs) ** 2))
        
        u_pred_bc_left, _, _ = AWPINN_model(x_bc_left, y_bc)
        u_pred_bc_right, _, _ = AWPINN_model(x_bc_right, y_bc)
        u_pred_bc_bottom, _, _ = AWPINN_model(x_bc, y_bc_bottom)
        u_pred_bc_top, _, _ = AWPINN_model(x_bc, y_bc_top)

        
        bc_loss = torch.mean((u_pred_bc_left - u_bc_left) ** 2) +\
                torch.mean((u_pred_bc_right - u_bc_right) ** 2) +\
                torch.mean((u_pred_bc_bottom - u_bc_bottom) ** 2) + \
                torch.mean((u_pred_bc_top - u_bc_top) ** 2)
        
        # Total loss with optional weighting
        total_loss = 0.005 * pde_loss + bc_loss
        
        return total_loss, pde_loss, bc_loss

    def train_awpinn(optimizer):

        global itr
        itr = 0

        def closure():
            global itr
            optimizer.zero_grad()
            total_loss, pde_loss, bc_loss = awpinn_loss()

            
            total_loss.backward()

            if itr % 1000 == 0 or itr == 4999 or itr == 100:

                numerical, _, _ = AWPINN_model(x_validation.to(device), y_validation.to(device))
                errL2 = (torch.sum(torch.abs(exact_validation.to(device)-numerical)**2))**0.5 / (torch.sum(torch.abs(exact_validation.to(device))**2))**0.5
                errMax = torch.max(torch.abs(exact_validation.to(device)-numerical))
                


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
        numerical, _, _ = AWPINN_model(x_test.to(device), y_test.to(device))
    torch.cuda.empty_cache()
    return numerical.cpu(), exact_test.cpu(), e
i = 0 # instance index
numerical, exact_test, e = train_awpinn_for_fno(i)
ErrL2 = (torch.sum(torch.abs(exact_test.reshape(-1)-numerical.reshape(-1))**2))**0.5 / (torch.sum(torch.abs(exact_test.reshape(-1))**2))**0.5
ErrMax = torch.max(torch.abs(exact_test.reshape(-1)-numerical.reshape(-1)))
    