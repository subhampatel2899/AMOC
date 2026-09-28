import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
# matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
from timeit import default_timer
from helmoltz.config import *  
from helmoltz.afpinns import AFPINN
# from helmoltz.parameter import *
from tqdm import tqdm

seed = 121
torch.manual_seed(seed)
np.random.seed(seed)
#load npz file
data = np.load('data/helm_pred_fno_out.npz')
u_fno_pred = torch.from_numpy(data['pred']).float()
y = torch.from_numpy(data['exact']).float()
data1 = np.load('data/helmholtz_test_out.npz')
b1_load = torch.from_numpy(data1['b1']).float()
b2_load = torch.from_numpy(data1['b2']).float()
b3_load = torch.from_numpy(data1['b3']).float()
k_load = torch.from_numpy(data1['k']).float()
u = torch.from_numpy(data1['u']).float()
Err = []
def train_afpinn_for_fno(i, n_test):
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
    u_vol_np    = u_vol.cpu().numpy() if isinstance(u_vol, torch.Tensor) else u_vol
    u_cosine_np = u_cosine.cpu().numpy()

    bias_init = float(np.mean(u_vol_np.reshape(-1) - u_cosine_np.reshape(-1)))
    kx    = torch.tensor(kx,  dtype=torch.float32)
    ky    = torch.tensor(ky,  dtype=torch.float32)   # was kt (spatial y)
    kz    = torch.tensor(kz,  dtype=torch.float32)   # NEW: z wave numbers
    phi   = torch.tensor(phi, dtype=torch.float32)
    coeff = torch.tensor(amp, dtype=torch.float32)

    AFPINN_model = AFPINN(kx, ky, kz, phi, coeff, bias=bias_init).to(device)

    optimizer_AFPINN = torch.optim.LBFGS(AFPINN_model.parameters(), 
                                lr=1.0,                            
                                max_iter=25000,
                                max_eval=10**10,
                                tolerance_grad=1e-10,
                                tolerance_change=1e-10,
                                history_size=50,
                                line_search_fn=None)
    x_interior = x_collocation.clone()
    y_interior = y_collocation.clone()
    z_interior = z_collocation.clone()
    ### Add b1 and b_2
    b1 = b1_load[i].to(device)
    b2 = b2_load[i].to(device)
    b3 = b3_load[i].to(device)
    k = k_load[i].to(device)
    def analytical(x, y, z):
        return torch.sin(2 * torch.pi * b1 * x) * \
            torch.sin(2 * torch.pi * b2 * y) * \
            torch.sin(2 * torch.pi * b3 * z)

    def right_side(x, y, z):
        u_val = analytical(x, y, z)

        freq_sq = b1**2 + b2**2 + b3**2
        lambda_val = (2 * torch.pi)**2 * freq_sq
        k_sq = k**2
        print("lambda_val:", lambda_val.item(), "k_sq:", k_sq.item())
        return (k_sq - lambda_val) * u_val



    u_bc_left = analytical(x_bc_left, y_bc, z_bc)
    u_bc_right = analytical(x_bc_right, y_bc, z_bc)
    u_bc_bottom = analytical(x_bc, y_bc_bottom, z_bc)
    u_bc_top = analytical(x_bc, y_bc_top, z_bc)
    u_bc_back = analytical(x_bc, y_bc, z_bc_back)
    u_bc_front = analytical(x_bc, y_bc, z_bc_front)

    rhs = right_side(x_collocation, y_collocation, z_collocation)
    # exact_validation = analytical(x_validation, y_validation, z_validation)
    exact_test = analytical(x_test, y_test, z_test).reshape(n_test, n_test, n_test).cpu()
    x_interior = x_collocation.clone()
    y_interior = y_collocation.clone()
    z_interior = z_collocation.clone()
    COLLOC_BATCH_SIZE = 10000   # tune down to 5_000 if still OOM
    BC_BATCH_SIZE     = 5000    # BC points per face; usually small, keep large
    TEST_BATCH_SIZE = 5000
    u = 0.01
    v = 10.0

    def afpinn_loss_batched():
        n_total = x_interior.shape[0]
        # v = min(10.0, 100.0 * (0.99 ** itr))
        # u = 0.1
        # ── 1. PDE loss — batched, backward per chunk ──────────────────
        pde_loss_accum = torch.tensor(0.0, device=device)

        for start in range(0, n_total, COLLOC_BATCH_SIZE):
            end = min(start + COLLOC_BATCH_SIZE, n_total)
            xb  = x_interior[start:end]
            yb  = y_interior[start:end]
            zb  = z_interior[start:end]
            rb  = rhs[start:end]

            u_out, u_xx, u_yy, u_zz = AFPINN_model(xb, yb, zb)
            res = u_xx + u_yy + u_zz + k**2 * u_out - rb         # Helmholtz residual

            chunk_loss = res.pow(2).mean() * ((end - start) / n_total)
            pde_loss_accum = pde_loss_accum + chunk_loss.detach()
            (u * chunk_loss).backward()                 # PDE weight = 0.01


        # (faces, interior_pts, bc_values) pairs
        bc_faces = [
            (x_bc_left, y_bc, z_bc, u_bc_left),
            (x_bc_right, y_bc, z_bc, u_bc_right),
            (x_bc, y_bc_bottom, z_bc, u_bc_bottom),
            (x_bc, y_bc_top, z_bc, u_bc_top),
            (x_bc, y_bc, z_bc_back, u_bc_back),
            (x_bc, y_bc, z_bc_front, u_bc_front),
        ]

        bc_loss_accum = torch.tensor(0.0, device=device)
        n_bc_total = bc_faces[0][0].shape[0]               # points per face

        for (xf, yf, zf, uf) in bc_faces:
            face_loss_accum = torch.tensor(0.0, device=device)

            for start in range(0, n_bc_total, BC_BATCH_SIZE):
                end  = min(start + BC_BATCH_SIZE, n_bc_total)
                xb   = xf[start:end]
                yb   = yf[start:end]
                zb   = zf[start:end]
                ub   = uf[start:end]

                u_pred, _, _, _ = AFPINN_model(xb, yb, zb)
                chunk_bc = (u_pred - ub).pow(2).mean() * ((end - start) / n_bc_total)
                face_loss_accum = face_loss_accum + chunk_bc.detach()
                (v * chunk_bc).backward()                      # accumulate grads

            bc_loss_accum = bc_loss_accum + face_loss_accum

        total_loss_val = u * pde_loss_accum + v * bc_loss_accum
        return total_loss_val, pde_loss_accum, bc_loss_accum


    def train_afpinn(optimizer):
        global itr
        itr = 0

        def closure():
            global itr
            optimizer.zero_grad()

            total_loss_val, pde_loss_val, bc_loss_val = afpinn_loss_batched()

            if itr % 1000 == 0 or itr == 100:
                with torch.no_grad():
                    x_flat = x_test.reshape(-1)
                    y_flat = y_test.reshape(-1)
                    z_flat = z_test.reshape(-1)
                    n_test = x_flat.shape[0]

                    outputs = []
                    for start in range(0, n_test, TEST_BATCH_SIZE):
                        end = min(start + TEST_BATCH_SIZE, n_test)
                        u_batch, _, _, _ = AFPINN_model(
                            x_flat[start:end],
                            y_flat[start:end],
                            z_flat[start:end],
                        )
                        outputs.append(u_batch.cpu())

                    numerical_cpu = torch.cat(outputs, dim=0)

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


    loss = train_afpinn(optimizer_AFPINN)

    with torch.no_grad():
        x_flat = x_test.reshape(-1)
        y_flat = y_test.reshape(-1)
        z_flat = z_test.reshape(-1)
        n_test = x_flat.shape[0]

        outputs = []
        for start in range(0, n_test, TEST_BATCH_SIZE):
            end = min(start + TEST_BATCH_SIZE, n_test)
            u_batch, _, _, _ = AFPINN_model(
                x_flat[start:end],
                y_flat[start:end],
                z_flat[start:end],
            )
            outputs.append(u_batch.cpu())

        numerical = torch.cat(outputs, dim=0)
        exact    = exact_test.reshape(-1)
    return numerical.cpu(), exact.cpu()


i = 0
numerical, exact = train_afpinn_for_fno(i, n_test)
ErrL2 = (torch.sum(torch.abs(exact.reshape(-1)-numerical.reshape(-1))**2))**0.5 / (torch.sum(torch.abs(exact.reshape(-1))**2))**0.5
ErrMax = torch.max(torch.abs(exact.reshape(-1)-numerical.reshape(-1)))
print(f'Test Relative L2 Error: {ErrL2.item()}, Test Max Error: {ErrMax.item()}')
    
