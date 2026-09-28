import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import default_timer
from poisson.config import *  # expects device, seed
from poisson.waveletfamily import *
from poisson.coeff_selection import *
from poisson.parameter import *
from poisson.awpinns import AWPINN
from tqdm import tqdm
import time
import random
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# #load npz file
data = np.load('data/poisson_pred_fno_out.npz', allow_pickle=True)
u_fno_pred = data['pred']
data = np.load('data/test_dataset_out.npz', allow_pickle=True)
params = data['params']

Jx = torch.arange(0, 6, device=device)
Jy = torch.arange(0, 6, device=device)
Jz = torch.arange(0, 6, device=device)

a = 0.3

len_family, family = wavelet_family(Jx, Jy, Jz, a)
print("family_len:", len_family)
jx, jy, jz = family[:,0], family[:,1], family[:,2]
kx, ky, kz = family[:,3], family[:,4], family[:,5]
x_vec, y_vec, z_vec = x_fno.reshape(-1), y_fno.reshape(-1), z_fno.reshape(-1)
dx = (x_upper - x_lower) / (n_fno - 1)
dy = (y_upper - y_lower) / (n_fno - 1)
dz = (z_upper - z_lower) / (n_fno - 1)
chunk_size = 256
Err = []
def train_awpinn_for_fno(i, n_test):
    torch.cuda.reset_peak_memory_stats(device)
    out = torch.from_numpy(u_fno_pred[i]).float().to(device)
    u_vec = out.squeeze(0).reshape(-1).to(device)
    
    coeff = apply_WT(u_vec,x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz) * dx * dy * dz
    coeff = conjugate_gradient(coeff, x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz, dx, dy, dz, max_iter=1, tol=1e-1)
    u_recon = apply_W(coeff, x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz)
    bias = torch.mean(u_recon - u_vec)

    top_indices = torch.nonzero(abs(coeff) > torch.quantile(abs(coeff), 0.99), as_tuple=True)[0]
    selected_indices = torch.unique(top_indices)
    coeff_new = coeff[selected_indices]
    family_new = family[selected_indices]

    print("coeff_len: ", len(coeff_new))
    print("Family len: ", len(family_new))

    wx = family_new[:,0].float().to(device)
    bx = family_new[:,3].float().to(device)
    wy = family_new[:,1].float().to(device)
    by = family_new[:,4].float().to(device)
    wt = family_new[:,2].float().to(device)
    bt = family_new[:,5].float().to(device)

    AWPINN_model = AWPINN(wx, bx, wy, by, wt, bt, coeff_new, bias).to(device)
    AWPINN_model = torch.compile(AWPINN_model, backend="eager")

    optimizer_AWPINN = torch.optim.LBFGS(AWPINN_model.parameters(), 
                                lr=1.0,                            
                                max_iter=5000,
                                max_eval=10**10,
                                tolerance_grad=1e-10,
                                tolerance_change=1e-10,
                                history_size=50,
                                line_search_fn=None)
    u_bc_left = analytical(x_bc_left, y_bc, z_bc, params[i])
    u_bc_right = analytical(x_bc_right, y_bc, z_bc, params[i])
    u_bc_bottom = analytical(x_bc, y_bc_bottom, z_bc, params[i])
    u_bc_top = analytical(x_bc, y_bc_top, z_bc, params[i])
    u_bc_back = analytical(x_bc, y_bc, z_bc_back, params[i])
    u_bc_front = analytical(x_bc, y_bc, z_bc_front, params[i])

    rhs = right_side(x_collocation, y_collocation, z_collocation, params[i])
    # exact_validation = analytical(x_validation, y_validation, z_validation, params)
    exact_test = analytical(x_test, y_test, z_test, params[i]).reshape(n_test, n_test, n_test).cpu()

    x_interior = x_collocation.clone()
    y_interior = y_collocation.clone()
    z_interior = z_collocation.clone()
    v = 0.01  # PDE loss weight
    # ──────────────────────────────────────────────────────────────────

    COLLOC_BATCH_SIZE = 5000   # tune down to 5_000 if still OOM
    BC_BATCH_SIZE     = 5000    # BC points per face; usually small, keep large
    TEST_BATCH_SIZE = 5000
    def awpinn_loss_batched():
        n_total = x_interior.shape[0]
        # ── 1. PDE loss — batched, backward per chunk ──────────────────
        pde_loss_accum = torch.tensor(0.0, device=device)

        for start in range(0, n_total, COLLOC_BATCH_SIZE):
            end = min(start + COLLOC_BATCH_SIZE, n_total)
            xb  = x_interior[start:end]
            yb  = y_interior[start:end]
            zb  = z_interior[start:end]
            rb  = rhs[start:end]

            u_out, u_xx, u_yy, u_zz = AWPINN_model(xb, yb, zb)
            res = -(u_xx + u_yy + u_zz + rb)         

            chunk_loss = res.pow(2).mean() * ((end - start) / n_total)
            pde_loss_accum = pde_loss_accum + chunk_loss.detach()
            (v * chunk_loss).backward()                 # PDE weight = 0.01


        # 2. BC loss — batched over each face
        bc_faces = [
            (x_bc_left, y_bc, z_bc, u_bc_left),
            (x_bc_right, y_bc, z_bc, u_bc_right),
            (x_bc, y_bc_bottom, z_bc, u_bc_bottom),
            (x_bc, y_bc_top, z_bc, u_bc_top),
            (x_bc, y_bc, z_bc_back, u_bc_back),
            (x_bc, y_bc, z_bc_front, u_bc_front),
        ]

        bc_loss_accum = torch.tensor(0.0, device=device)
        n_bc_total = bc_faces[0][0].shape[0]               

        for (xf, yf, zf, uf) in bc_faces:
            face_loss_accum = torch.tensor(0.0, device=device)

            for start in range(0, n_bc_total, BC_BATCH_SIZE):
                end  = min(start + BC_BATCH_SIZE, n_bc_total)
                xb   = xf[start:end]
                yb   = yf[start:end]
                zb   = zf[start:end]
                ub   = uf[start:end]

                u_pred, _, _, _ = AWPINN_model(xb, yb, zb)
                chunk_bc = (u_pred - ub).pow(2).mean() * ((end - start) / n_bc_total)
                face_loss_accum = face_loss_accum + chunk_bc.detach()
                (chunk_bc).backward()                      # accumulate grads

            bc_loss_accum = bc_loss_accum + face_loss_accum

        total_loss_val = v * pde_loss_accum + bc_loss_accum
        return total_loss_val, pde_loss_accum, bc_loss_accum


    def train_awpinn(optimizer):
        global itr
        itr = 0

        def closure():
            global itr
            optimizer.zero_grad()

            total_loss_val, pde_loss_val, bc_loss_val = awpinn_loss_batched()

            if (itr + 1) % 1000 == 0 or itr == 0:
                with torch.no_grad():
                    x_flat = x_test.reshape(-1)
                    y_flat = y_test.reshape(-1)
                    z_flat = z_test.reshape(-1)
                    n_test = x_flat.shape[0]

                    outputs = []
                    for start in range(0, n_test, TEST_BATCH_SIZE):
                        end = min(start + TEST_BATCH_SIZE, n_test)
                        u_batch, _, _, _ = AWPINN_model(
                            x_flat[start:end],
                            y_flat[start:end],
                            z_flat[start:end],
                            compute_derivatives=False
                        )
                        outputs.append(u_batch.cpu())

                    numerical_cpu = torch.cat(outputs, dim=0)

                    # print("numerical shape:", numerical_cpu.shape, "exact shape:", exact_test.shape)
                    errL2 = (
                        torch.sum((exact_test.reshape(-1) - numerical_cpu).abs() ** 2) ** 0.5
                        / torch.sum(exact_test.reshape(-1).abs() ** 2) ** 0.5
                    )
                    errMax = torch.max((exact_test.reshape(-1) - numerical_cpu).abs())

                print(
                    f'Epoch[{itr}]  '
                    f'Total Loss: {total_loss_val.item():.6f}, '
                    f'PDE Loss: {pde_loss_val.item():.6f}, '
                    f'BC Loss: {bc_loss_val.item():.6f}\n\t\t'
                    f'RelativeL2: {errL2:.6f},\t'
                    f'Max: {errMax:.6f}\n'
                )
                torch.cuda.empty_cache()

            itr += 1
            return total_loss_val

        loss = optimizer.step(closure)
        return loss


    loss = train_awpinn(optimizer_AWPINN)

    with torch.no_grad():
        x_flat = x_test.reshape(-1)
        y_flat = y_test.reshape(-1)
        z_flat = z_test.reshape(-1)
        n_test = x_flat.shape[0]

        outputs = []
        for start in range(0, n_test, TEST_BATCH_SIZE):
            end = min(start + TEST_BATCH_SIZE, n_test)
            u_batch, _, _, _ = AWPINN_model(
                x_flat[start:end],
                y_flat[start:end],
                z_flat[start:end],
                compute_derivatives=False
            )
            outputs.append(u_batch.cpu())

        numerical_cpu = torch.cat(outputs, dim=0)

    return numerical_cpu.cpu(), exact_test.cpu()

for i in tqdm(range(1)):
    numerical, exact_test = train_awpinn_for_fno(1, n_test)


ErrL2 = (torch.sum((exact_test.reshape(-1) - numerical.reshape(-1)) ** 2) ** 0.5 /
            torch.sum(exact_test.reshape(-1) ** 2) ** 0.5)
ErrMax = torch.max(torch.abs(exact_test.reshape(-1) - numerical.reshape(-1)))

peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

print(f"FINAL_L2: {ErrL2.item():.6f}")
print(f"FINAL_MAX: {ErrMax.item():.6f}")
print(f"FINAL_MEM: {peak_mem_mb:.2f}")