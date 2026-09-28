import torch
import torch.nn as nn
from poisson.config import *
from tqdm import tqdm
from poisson.waveletfamily import *

def apply_W(v,x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz, chunk_size=2**10):
    out = torch.zeros_like(x_vec)

    for i in tqdm(range(0, len(jx), chunk_size)):
        W_chunk = gaussian(
            x_vec, y_vec, z_vec,
            jx[i:i+chunk_size],
            jy[i:i+chunk_size],
            jz[i:i+chunk_size],
            kx[i:i+chunk_size],
            ky[i:i+chunk_size],
            kz[i:i+chunk_size]
        )   

        out += W_chunk.T @ v[i:i+chunk_size]
        del W_chunk

    return out


def apply_WT(u, x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz, chunk_size=2**10):
    coeff = torch.zeros(len(jx), device=device)

    for i in tqdm(range(0, len(jx), chunk_size)):
        W_chunk = gaussian(
            x_vec, y_vec, z_vec,
            jx[i:i+chunk_size],
            jy[i:i+chunk_size],
            jz[i:i+chunk_size],
            kx[i:i+chunk_size],
            ky[i:i+chunk_size],
            kz[i:i+chunk_size]
        )

        coeff[i:i+chunk_size] = W_chunk @ u
        del W_chunk

    return coeff


lambda_reg = 1e-6

def apply_Gram(v, x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz, dx, dy, dz):
    return apply_WT(apply_W(v, x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz), x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz) * dx * dy * dz + lambda_reg * v


def conjugate_gradient(b, x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz, dx, dy, dz, max_iter=1, tol=1e-2):

    x = torch.zeros_like(b)
    r = b - apply_Gram(x, x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz, dx, dy, dz)
    p = r.clone()

    rs_old = torch.dot(r, r)

    for i in tqdm(range(max_iter)):
        Ap = apply_Gram(p, x_vec, y_vec, z_vec, jx, jy, jz, kx, ky, kz, dx, dy, dz)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)

        x += alpha * p
        r -= alpha * Ap

        rs_new = torch.dot(r, r)
        rel_residual = torch.sqrt(rs_new) / torch.norm(b)
        print(f"Iter {i+1}: Relative Residual = {rel_residual.item():.4e}")

        if rel_residual < tol:
            print(f"CG converged in {i} iterations")
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x