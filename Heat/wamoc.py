import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
# matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
from timeit import default_timer
from heat.config import *  # expects device, seed
from heat.waveletfamily import *
from heat.coeff_selection import *
from heat.awpinns import AWPINN
from tqdm import tqdm

seed = 121
torch.manual_seed(seed)
np.random.seed(seed)

test_data = np.load("data/heat_data_out.npz", allow_pickle=True)
eps_test = torch.tensor(test_data["eps"], dtype=torch.float32)
u_test = torch.tensor(test_data["u"], dtype=torch.float32)
pred_data = np.load("data/heat_pred_fno_out.npz")
u_fno_pred = torch.tensor(pred_data["pred"], dtype=torch.float32)

dx = (x_fno[-1] - x_fno[0])/(n_fno-1)
dy = (t_fno[-1] - t_fno[0])/(n_fno-1)
Jx = torch.arange(0, 6)
Jy = torch.arange(0, 6)
a = 0.2
family = wavelet_family(Jx, Jy, a, x_lower, x_upper, t_lower, t_upper)
jx, jy, kx, ky = family[:,0], family[:,1], family[:,2], family[:,3]
# dx.shape
chunk_size = 256
Err = []
def train_awpinn_for_fno(i):
    u_vec = u_fno_pred[i].reshape(-1).to(device)

    coeff = apply_WT(u_vec, x_fno.to(device), t_fno.to(device), jx, jy, kx, ky, chunk_size=chunk_size) * dx * dy
    coeff = conjugate_gradient(coeff, dx, dy, x_fno.to(device), t_fno.to(device), jx, jy, kx, ky, max_iter=1, tol=1e-1)


    u_recon_vec = apply_W(coeff, x_fno.to(device), t_fno.to(device), jx, jy, kx, ky, chunk_size=chunk_size)
    bias = (u_recon_vec - u_vec).mean()
    print("Bias: ", bias.item())
    
    top_indices = torch.nonzero(abs(coeff) > torch.quantile(abs(coeff), 0.90), as_tuple=True)[0]

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



    def awpinn_loss():   

        u, u_t, u_xx = AWPINN_model(x_interior, t_interior)
        
        u_pred_ic_left, _, _ = AWPINN_model(x_ic, t_ic)
        u_pred_bc_left, _, _ = AWPINN_model(x_bc_left, t_bc)
        u_pred_bc_right, _, _ = AWPINN_model(x_bc_right, t_bc)

        pde_loss = torch.mean((u_t - u_xx - rhs) ** 2)
        
        ic_loss = torch.mean((u_pred_ic_left - u_ic)** 2)

        bc_loss = torch.mean((u_pred_bc_left - u_bc_left) ** 2) + \
                torch.mean((u_pred_bc_right - u_bc_right) ** 2)
        

        total_loss = 0.01 * pde_loss + ic_loss + bc_loss
        
        return total_loss, pde_loss, ic_loss, bc_loss

    def train_awpinn(optimizer):

        global itr
        itr = 0

        def closure():
            global itr
            optimizer.zero_grad()
            total_loss, pde_loss, ic_loss, bc_loss = awpinn_loss()

            
            total_loss.backward()

            if itr % 1000 == 0 or itr == 4999:

                numerical, _, _ = AWPINN_model(x_validation.to(device), t_validation.to(device))
                errL2 = (torch.sum(torch.abs(exact_validation.to(device)-numerical)**2))**0.5 / (torch.sum(torch.abs(exact_validation.to(device))**2))**0.5
                errMax = torch.max(torch.abs(exact_validation.to(device)-numerical))
                


                print(f'Epoch[{itr}]  '
                    f'Total Loss: {total_loss.item():.6f}, '
                        f'PDE Loss: {pde_loss.item():.6f}, '
                        f'IC Loss: {ic_loss.item():.6f}, '
                        f'BC Loss: {bc_loss.item():.6f}\n\t\t'
                        f'RelativeL2: {errL2},\t'
                        f'Max: {errMax}\n' )
                
                torch.cuda.empty_cache()
                if torch.isnan(total_loss):
                    print("NaN loss encountered. Stopping training.")
                    return total_loss
            
            itr += 1


            return total_loss

        loss = optimizer.step(closure)

    loss = train_awpinn(optimizer_AWPINN)

    with torch.no_grad():
        numerical, _, _ = AWPINN_model(x_test.to(device), t_test.to(device))
    torch.cuda.empty_cache()
    return numerical.cpu(), exact_test.cpu(), e
    
    
i = 0 #instance index
numerical, exact_test, e = train_awpinn_for_fno(i)
ErrL2 = (torch.sum(torch.abs(exact_test.reshape(-1)-numerical.reshape(-1))**2))**0.5 / (torch.sum(torch.abs(exact_test.reshape(-1))**2))**0.5
ErrMax = torch.max(torch.abs(exact_test.reshape(-1)-numerical.reshape(-1)))
    