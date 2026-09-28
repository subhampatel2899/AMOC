
import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
# matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
from timeit import default_timer
from poisson.config import *  
from poisson.afpinns import AFPINN
from poisson.parameter import *
from tqdm import tqdm

seed = 121
torch.manual_seed(seed)
np.random.seed(seed)
# #load npz file
data = np.load('data/poisson_pred_fno_in.npz', allow_pickle=True)
u_fno_pred = data['pred']
data = np.load('data/test_dataset_in.npz', allow_pickle=True)
params = data['params']
Err = []
def train_afpinn_for_fno(i):
    N = n_fno
    K = 200
    u_vol = u_fno_pred[i].reshape(N, N, N)  # full 3D volume
    U_fft = np.fft.fftn(u_vol)                          # 3D FFT

    amplitudes = np.abs(U_fft)
    top_k_flat = np.argpartition(amplitudes.ravel(), -K)[-K:]
    top_k_idx  = np.array(np.unravel_index(top_k_flat, (N, N, N))).T
    order      = np.argsort(-amplitudes[top_k_idx[:,0], top_k_idx[:,1], top_k_idx[:,2]])
    top_k_idx  = top_k_idx[order]

    ix_sel, iy_sel, iz_sel = top_k_idx[:,0], top_k_idx[:,1], top_k_idx[:,2]
    C_sel = U_fft[ix_sel, iy_sel, iz_sel]

    freq = np.fft.fftfreq(N) * N
    ix_s = freq[ix_sel];  iy_s = freq[iy_sel];  iz_s = freq[iz_sel]

    x_min = float(x_fno.min()); y_min = float(y_fno.min()); z_min = float(z_fno.min())
    Lx = float(x_fno.max() - x_fno.min())
    Ly = float(y_fno.max() - y_fno.min())
    Lz = float(z_fno.max() - z_fno.min())

    kx_sel  = 2 * np.pi * ix_s / Lx
    ky_sel  = 2 * np.pi * iy_s / Ly
    kz_sel  = 2 * np.pi * iz_s / Lz
    phi_sel = (-kx_sel*x_min - ky_sel*y_min - kz_sel*z_min + np.angle(C_sel)).astype(np.float32)
    amp_sel = (np.abs(C_sel) / N**3).astype(np.float32)

    # Remove DC component
    mask = ~((ix_sel==0) & (iy_sel==0) & (iz_sel==0))
    kx, ky, kz = kx_sel[mask].astype(np.float32), ky_sel[mask].astype(np.float32), kz_sel[mask].astype(np.float32)
    phi, amp   = phi_sel[mask], amp_sel[mask]
    K_eff = int(mask.sum())

    # Reconstruction check
    X = x_fno.reshape(N,N,N); Y = y_fno.reshape(N,N,N); Z = z_fno.reshape(N,N,N)
    u_cosine = sum(amp[k]*torch.cos(kx[k]*X + ky[k]*Y + kz[k]*Z + phi[k]) for k in range(K_eff))
    bias_init = float(np.mean(u_vol - u_cosine.cpu().numpy()))
    kx    = torch.tensor(kx,  dtype=torch.float32)
    ky    = torch.tensor(ky,  dtype=torch.float32)   # was kt (spatial y)
    kz    = torch.tensor(kz,  dtype=torch.float32)   # NEW: z wave numbers
    phi   = torch.tensor(phi, dtype=torch.float32)
    coeff = torch.tensor(amp, dtype=torch.float32)

    AFPINN_model = AFPINN(kx, ky, kz, phi, coeff, bias=bias_init).to(device)

    x_interior = x_collocation.clone()
    y_interior = y_collocation.clone()
    z_interior = z_collocation.clone()
    n_test = len(x_test)
    u_bc_left = analytical(x_bc_left, y_bc, z_bc, params[i])
    u_bc_right = analytical(x_bc_right, y_bc, z_bc, params[i])
    u_bc_bottom = analytical(x_bc, y_bc_bottom, z_bc, params[i])
    u_bc_top = analytical(x_bc, y_bc_top, z_bc, params[i])
    u_bc_back = analytical(x_bc, y_bc, z_bc_back, params[i])
    u_bc_front = analytical(x_bc, y_bc, z_bc_front, params[i])

    rhs = right_side(x_collocation, y_collocation, z_collocation, params[i])
    # exact_validation = analytical(x_validation, y_validation, z_validation, params)
    exact_test = analytical(x_test, y_test, z_test, params[i]).reshape(n_test, n_test, n_test).cpu()
    # ── Hyperparameters ────────────────────────────────────────────────
    v_pde        = 0.01      
    v_bc         = 1.0    
    COLLOC_BATCH_SIZE = 10000
    BC_BATCH_SIZE     = 5000
    TEST_BATCH_SIZE   = 5000

    # ── Boundary values ───────────────────────────────────────────────
    u_bc_left   = analytical(x_bc_left,  y_bc,       z_bc,       params[i])
    u_bc_right  = analytical(x_bc_right, y_bc,       z_bc,       params[i])
    u_bc_bottom = analytical(x_bc,       y_bc_bottom, z_bc,      params[i])
    u_bc_top    = analytical(x_bc,       y_bc_top,    z_bc,      params[i])
    u_bc_back   = analytical(x_bc,       y_bc,       z_bc_back,  params[i])
    u_bc_front  = analytical(x_bc,       y_bc,       z_bc_front, params[i])

    rhs         = right_side(x_collocation, y_collocation, z_collocation, params[i])
    exact_test  = analytical(x_test, y_test, z_test, params[i]).cpu()  # shape (64,64,64)

    x_interior = x_collocation.clone()
    y_interior = y_collocation.clone()
    z_interior = z_collocation.clone()


    optimizer_AFPINN = torch.optim.LBFGS(
        AFPINN_model.parameters(),
        lr=1.0,
        max_iter=5000,           # per step, not total
        max_eval=10**10,
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
        history_size=50,       # larger for 3D parameter count
        line_search_fn=None   # FIX: was None
    )


    def afpinn_loss_batched():
        n_total = x_interior.shape[0]

        # keep one live (non-detached) handle for LBFGS curvature
        pde_loss_accum  = torch.tensor(0.0, device=device)
        bc_loss_accum   = torch.tensor(0.0, device=device)

        # ── PDE loss ──────────────────────────────────────────────────
        for start in range(0, n_total, COLLOC_BATCH_SIZE):
            end  = min(start + COLLOC_BATCH_SIZE, n_total)
            xb, yb, zb, rb = (x_interior[start:end], y_interior[start:end],
                            z_interior[start:end], rhs[start:end])

            u_out, u_xx, u_yy, u_zz = AFPINN_model(xb, yb, zb)
            res        = -(u_xx + u_yy + u_zz) - rb
            chunk_loss = res.pow(2).mean() * ((end - start) / n_total)
            pde_loss_accum   = pde_loss_accum + chunk_loss          # stays in graph
            (v_pde * chunk_loss).backward(retain_graph=False)

        # ── BC loss — 6 faces ─────────────────────────────────────────
        bc_faces = [
            (x_bc_left,  y_bc,        z_bc,        u_bc_left),
            (x_bc_right, y_bc,        z_bc,        u_bc_right),
            (x_bc,       y_bc_bottom, z_bc,        u_bc_bottom),
            (x_bc,       y_bc_top,    z_bc,        u_bc_top),
            (x_bc,       y_bc,        z_bc_back,   u_bc_back),
            (x_bc,       y_bc,        z_bc_front,  u_bc_front),
        ]
        n_bc_total = bc_faces[0][0].shape[0]

        for (xf, yf, zf, uf) in bc_faces:
            for start in range(0, n_bc_total, BC_BATCH_SIZE):
                end      = min(start + BC_BATCH_SIZE, n_bc_total)
                u_pred, _, _, _ = AFPINN_model(xf[start:end], yf[start:end], zf[start:end])
                chunk_bc = (u_pred - uf[start:end]).pow(2).mean() * ((end - start) / n_bc_total)
                bc_loss_accum  = bc_loss_accum + chunk_bc           # stays in graph
                (v_bc * chunk_bc).backward(retain_graph=False)

        total_loss_val = v_pde * pde_loss_accum.detach() + v_bc * bc_loss_accum.detach()
        return total_loss_val, pde_loss_accum.detach(), bc_loss_accum.detach()


    def train_afpinn(optimizer):
        global itr
        itr = 0

        def closure():
            global itr
            optimizer.zero_grad()
            total_loss_val, pde_loss_val, bc_loss_val = afpinn_loss_batched()

            if itr % 1000 == 0:
                with torch.no_grad():
                    x_flat = x_test.reshape(-1)    
                    y_flat = y_test.reshape(-1)
                    z_flat = z_test.reshape(-1)
                    n_pts  = x_flat.shape[0]       

                    outputs = []
                    for start in range(0, n_pts, TEST_BATCH_SIZE):
                        end = min(start + TEST_BATCH_SIZE, n_pts)
                        u_batch, _, _, _ = AFPINN_model(
                            x_flat[start:end], y_flat[start:end], z_flat[start:end]
                        )
                        outputs.append(u_batch.cpu())

                    numerical_cpu = torch.cat(outputs, dim=0)
                    exact_flat    = exact_test.reshape(-1)

                    errL2  = (torch.sum((exact_flat - numerical_cpu).abs()**2)**0.5
                            / torch.sum(exact_flat.abs()**2)**0.5)
                    errMax = torch.max((exact_flat - numerical_cpu).abs())

                print(
                    f'Epoch[{itr}]  '
                    f'Total Loss: {total_loss_val.item():.6f}, '
                    f'PDE Loss: {pde_loss_val.item():.6f}, '
                    f'BC Loss: {bc_loss_val.item():.6f}\n\t\t'
                    f'RelativeL2: {errL2:.6f},\tMax: {errMax:.6f}\n'
                )
            torch.cuda.empty_cache()

            itr += 1
            return total_loss_val

        return optimizer.step(closure)


    loss = train_afpinn(optimizer_AFPINN)
    with torch.no_grad():
        x_flat = x_test.reshape(-1)    
        y_flat = y_test.reshape(-1)
        z_flat = z_test.reshape(-1)
        n_pts  = x_flat.shape[0]       

        outputs = []
        for start in range(0, n_pts, TEST_BATCH_SIZE):
            end = min(start + TEST_BATCH_SIZE, n_pts)
            u_batch, _, _, _ = AFPINN_model(
                x_flat[start:end], y_flat[start:end], z_flat[start:end]
            )
            outputs.append(u_batch.cpu())

        numerical = torch.cat(outputs, dim=0)
        exact    = exact_test.reshape(-1)
    return numerical.cpu(), exact.cpu()

i = 0 #instance index
numerical, exact = train_afpinn_for_fno(i)
ErrL2 = (torch.sum(torch.abs(exact.reshape(-1)-numerical.reshape(-1))**2))**0.5 / (torch.sum(torch.abs(exact.reshape(-1))**2))**0.5
ErrMax = torch.max(torch.abs(exact.reshape(-1)-numerical.reshape(-1)))