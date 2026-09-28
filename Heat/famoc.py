import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
# matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
from timeit import default_timer
from heat.config import *  
from heat.afpinns import AFPINN
from tqdm import tqdm

seed = 121
torch.manual_seed(seed)
np.random.seed(seed)
# Data Loader
test_data = np.load("data/heat_data_in.npz", allow_pickle=True)
eps_test = torch.tensor(test_data["eps"], dtype=torch.float32)
u_test = torch.tensor(test_data["u"], dtype=torch.float32)
pred_data = np.load("data/heat_pred_fno_in.npz")
u_fno_pred = torch.tensor(pred_data["pred"], dtype=torch.float32)
# mode
N = n_fno
K = 200
x_min, x_max = float(x_fno.min().item()), float(x_fno.max().item())
t_min, t_max = float(t_fno.min().item()), float(t_fno.max().item())
Lx = (x_max - x_min)
Lt = (t_max - t_min)
chunk_size = 256
Err = []
def train_afpinn_for_fno(i):
    u_grid = u_fno_pred[i,:,:].cpu().numpy()  # shape (N, N)
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
    kt_sel  = 2 * np.pi * it_s / Lt
    phi_sel = -kx_sel * x_min - kt_sel * t_min + np.angle(C_sel)
    amp_sel = np.abs(C_sel) / (N**2)    


    mask = ~((ix_sel==0) & (it_sel==0))
    kx = kx_sel[mask].astype(np.float32)
    kt = kt_sel[mask].astype(np.float32)
    phi = phi_sel[mask].astype(np.float32)
    amp = amp_sel[mask].astype(np.float32)
    K_eff = int(mask.sum())

    u_cosine_sum = sum(
        amp[k] * torch.cos(kx[k]*x_fno.reshape(n_fno,n_fno) + kt[k]*t_fno.reshape(n_fno,n_fno) + phi[k])
        for k in range(K_eff)
    )
    bias_init = float(np.mean(u_grid - u_cosine_sum.numpy()))
    u_recon   = u_cosine_sum + bias_init

    err_numpy = np.linalg.norm(u_recon - u_grid) / np.linalg.norm(u_grid)
    print(f"K_eff={K_eff}, bias={bias_init:.6f}")
    print(f"Numpy physical reconstruction rel-L2: {err_numpy:.4e}")

    kx, kt,phi, coeff = torch.tensor(kx), torch.tensor(kt), torch.tensor(phi), torch.tensor(amp)
    AFPINN_model = AFPINN(kx, kt, phi, coeff, bias = bias_init).to(device)

    optimizer_AFPINN = torch.optim.LBFGS(AFPINN_model.parameters(), 
                                lr=1.0,                            
                                max_iter=5*10**3,
                                max_eval=10**10,
                                tolerance_grad=1e-9,
                                tolerance_change=1e-9,
                                history_size=50,
                                line_search_fn=None)




    x_interior = x_collocation.clone()
    t_interior = t_collocation.clone()
    e = eps_test[i,0,0].cpu().item()
    def analytical(x,t):
        et = 2*t-1
        ex = torch.exp(1/(et**2 + e))

        return (1-x**2)*ex


    def right_side(x,t):
        et = 2*t-1
        ex = torch.exp(1/(et**2 + e))

        return 2*ex*(1 + 2*et*(x**2-1)/(et**2+e)**2)


    u_ic = analytical(x_ic, t_ic)
    u_bc_left = analytical(x_bc_left, t_bc)
    u_bc_right = analytical(x_bc_right, t_bc)

    rhs = right_side(x_collocation, t_collocation)
    exact_validation = analytical(x_validation, t_validation)
    exact_test = analytical(x_test, t_test).reshape(n_test, n_test)



    def afpinn_loss():   

        u, u_t, u_xx = AFPINN_model(x_interior, t_interior)
        
        u_pred_ic_left, _, _ = AFPINN_model(x_ic, t_ic)
        u_pred_bc_left, _, _ = AFPINN_model(x_bc_left, t_bc)
        u_pred_bc_right, _, _ = AFPINN_model(x_bc_right, t_bc)

        pde_loss = torch.mean((u_t - u_xx - rhs) ** 2)
        
        ic_loss = torch.mean((u_pred_ic_left - u_ic)** 2)

        bc_loss = torch.mean((u_pred_bc_left - u_bc_left) ** 2) + \
                torch.mean((u_pred_bc_right - u_bc_right) ** 2)
        
        # Total loss with optional weighting
        total_loss = 0.01 * pde_loss + ic_loss + bc_loss
        
        return total_loss, pde_loss, ic_loss, bc_loss

    def train_afpinn(optimizer):

        global itr
        itr = 0

        def closure():
            global itr
            optimizer.zero_grad()
            total_loss, pde_loss, ic_loss, bc_loss = afpinn_loss()

            # numerical, _, _ = AFPINN_model(x_validation.to(device), y_validation.to(device))
            # errL2 = (torch.sum(torch.abs(exact_validation.to(device)-numerical)**2))**0.5 / (torch.sum(torch.abs(exact_validation.to(device))**2))**0.5
            # errMax = torch.max(torch.abs(exact_validation.to(device)-numerical))

            
            total_loss.backward()

            if itr % 1000 == 0 or itr == 4999:

                numerical, _, _ = AFPINN_model(x_validation.to(device), t_validation.to(device))
                errL2 = (torch.sum(torch.abs(exact_validation.to(device)-numerical)**2))**0.5 / (torch.sum(torch.abs(exact_validation.to(device))**2))**0.5
                errMax = torch.max(torch.abs(exact_validation.to(device)-numerical))
                if torch.isnan(total_loss):
                    print("NaN loss encountered. Stopping training.")
                    return total_loss


                print(f'Epoch[{itr}]  '
                    f'Total Loss: {total_loss.item():.6f}, '
                        f'PDE Loss: {pde_loss.item():.6f}, '
                        f'IC Loss: {ic_loss.item():.6f}, '
                        f'BC Loss: {bc_loss.item():.6f}\n\t\t'
                        f'RelativeL2: {errL2},\t'
                        f'Max: {errMax}\n' )
                
                torch.cuda.empty_cache()
            
            itr += 1


            return total_loss

        loss = optimizer.step(closure)

    loss = train_afpinn(optimizer_AFPINN)

    with torch.no_grad():
        numerical, _, _ = AFPINN_model(x_test.to(device), t_test.to(device))
    torch.cuda.empty_cache()
    return numerical.cpu(), exact_test.cpu(), e

i = 0 #instance index
numerical, exact_test, e = train_afpinn_for_fno(i)
ErrL2 = (torch.sum(torch.abs(exact_test.reshape(-1)-numerical.reshape(-1))**2))**0.5 / (torch.sum(torch.abs(exact_test.reshape(-1))**2))**0.5
ErrMax = torch.max(torch.abs(exact_test.reshape(-1)-numerical.reshape(-1)))
