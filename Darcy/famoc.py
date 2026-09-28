import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
# matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
from timeit import default_timer
from darcy.config import *  # expects device, seed
from darcy.parameter import *
from darcy.afpinns import AFPINN
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
# mode
N = n_fno
K = 200
x_min, x_max = float(x_fno.min().item()), float(x_fno.max().item())
y_min, y_max = float(y_fno.min().item()), float(y_fno.max().item())
Lx = (x_max - x_min)
Ly = (y_max - y_min)
chunk_size = 256
Err = []
def train_afpinn_for_fno(i):
    u_grid = p_fno_pred[i].cpu().numpy()  # shape (N, N)
    U_fft  = np.fft.fft2(u_grid)
    amplitudes = np.abs(U_fft)
    top_k_flat = np.argpartition(amplitudes.ravel(), -K)[-K:]
    top_k_idx = np.array(np.unravel_index(top_k_flat, (N,N))).T
    order = np.argsort(-amplitudes[top_k_idx[:,0], top_k_idx[:,1]])
    top_k_idx = top_k_idx[order]

    ix_sel = top_k_idx[:,0]
    it_sel = top_k_idx[:,1]
    C_sel  = U_fft[ix_sel, it_sel]


    ix_signed = np.fft.fftfreq(N) * N
    it_signed = np.fft.fftfreq(N) * N

    ix_s = ix_signed[ix_sel]
    it_s = it_signed[it_sel]

    kx_sel  = 2 * np.pi   * ix_s / Lx
    kt_sel  = 2 * np.pi * it_s / Ly
    phi_sel = -kx_sel * x_min - kt_sel * y_min + np.angle(C_sel)
    amp_sel = np.abs(C_sel) / (N**2)    


    mask = ~((ix_sel==0) & (it_sel==0))
    kx = kx_sel[mask].astype(np.float32)
    kt = kt_sel[mask].astype(np.float32)
    phi = phi_sel[mask].astype(np.float32)
    amp = amp_sel[mask].astype(np.float32)
    K_eff = int(mask.sum())

    u_cosine_sum = sum(
        amp[k] * torch.cos(kx[k]*x_fno.reshape(n_fno,n_fno) + kt[k]*y_fno.reshape(n_fno,n_fno) + phi[k])
        for k in range(K_eff)
    )
    bias_init = float(np.mean(u_grid - u_cosine_sum.cpu().numpy()))
    u_recon   = u_cosine_sum + bias_init

    err_numpy = np.linalg.norm(u_recon.cpu() - u_grid) / np.linalg.norm(u_grid)
    print(f"K_eff={K_eff}, bias={bias_init:.6f}")
    print(f"Numpy physical reconstruction rel-L2: {err_numpy:.4e}")

    kx, kt,phi, coeff = torch.tensor(kx), torch.tensor(kt), torch.tensor(phi), torch.tensor(amp)
    AFPINN_model = AFPINN(kx, kt, phi, coeff, bias = bias_init).to(device)

    optimizer_AFPINN = torch.optim.LBFGS(AFPINN_model.parameters(), 
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



    def afpinn_loss():   

        u, u_x, u_y, u_xx, u_yy = AFPINN_model(x_interior, y_interior)
        # print(u.shape, u_x.shape, u_y.shape, u_xx.shape, u_yy.shape, k_p1.shape, kx_p1.shape, ky_p1.shape, rhs.shape)

        pde_loss = (torch.mean((- (k_p * (u_yy + u_xx) + dk_dx_p * u_x + dk_dy_p * u_y) - rhs)**2))
        
        u_pred_bc_left, _, _, _, _ = AFPINN_model(x_bc_left, y_bc)
        u_pred_bc_right, _, _, _, _ = AFPINN_model(x_bc_right, y_bc)
        u_pred_bc_bottom, _, _, _, _ = AFPINN_model(x_bc, y_bc_bottom)
        u_pred_bc_top, _, _, _, _ = AFPINN_model(x_bc, y_bc_top)

        
        bc_loss = torch.mean((u_pred_bc_left - u_bc_left) ** 2) +\
                torch.mean((u_pred_bc_right - u_bc_right) ** 2) +\
                torch.mean((u_pred_bc_bottom - u_bc_bottom) ** 2) + \
                torch.mean((u_pred_bc_top - u_bc_top) ** 2)
        
        # Total loss with optional weighting
        total_loss = pde_loss + 500 * bc_loss
        return total_loss, pde_loss, bc_loss

    def train_afpinn(optimizer):

        global itr
        itr = 0

        def closure():
            global itr
            optimizer.zero_grad()
            total_loss, pde_loss, bc_loss = afpinn_loss()

            total_loss.backward()

            if itr % 1000 == 0 or itr == 100:
                with torch.no_grad():

                    numerical, _, _, _, _ = AFPINN_model(x_test.to(device), y_test.to(device))
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

    loss = train_afpinn(optimizer_AFPINN)
    with torch.no_grad():
        numerical, _, _, _, _ = AFPINN_model(x_test.to(device), y_test.to(device))
    torch.cuda.empty_cache()
    return numerical.cpu(), exact.cpu()

i = 0 #instance index
numerical, exact = train_afpinn_for_fno(i)
ErrL2 = (torch.sum(torch.abs(exact.reshape(-1)-numerical.reshape(-1))**2))**0.5 / (torch.sum(torch.abs(exact.reshape(-1))**2))**0.5
ErrMax = torch.max(torch.abs(exact.reshape(-1)-numerical.reshape(-1)))
    