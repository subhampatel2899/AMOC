import torch
import numpy as np
from poisson.config import *
from poisson.waveletfamily import *


def apply_W(v, x_vec, y_vec, jx, jy, kx, ky, chunk_size=128):
    out = torch.zeros_like(x_vec)
    
    for i in range(0, len(jx), chunk_size):
        W_chunk = gaussian(
            x_vec, y_vec,
            jx[i:i+chunk_size],
            jy[i:i+chunk_size],
            kx[i:i+chunk_size],
            ky[i:i+chunk_size]
        )

        out += W_chunk.T @ v[i:i+chunk_size]
        del W_chunk

    return out


def apply_WT(u, x_vec, y_vec, jx, jy, kx, ky, chunk_size=128):
    coeff = torch.zeros(len(jx), device=device)
        
    for i in range(0, len(jx), chunk_size):
        W_chunk = gaussian(
            x_vec, y_vec,
            jx[i:i+chunk_size],
            jy[i:i+chunk_size],
            kx[i:i+chunk_size],
            ky[i:i+chunk_size]
        )

        coeff[i:i+chunk_size] = W_chunk @ u
        del W_chunk

    return coeff


# ===================== GRAM OPERATOR =====================
lambda_reg = 1e-6

def apply_Gram(v, dx, dy, x_vec, y_vec, jx, jy, kx, ky, chunk_size=128):
    return apply_WT(apply_W(v, x_vec, y_vec, jx, jy, kx, ky, chunk_size=chunk_size), x_vec, y_vec, jx, jy, kx, ky, chunk_size=chunk_size) * dx * dy + lambda_reg * v


# ===================== CONJUGATE GRADIENT =====================
def conjugate_gradient(b, dx, dy, x_vec, y_vec, jx, jy, kx, ky, max_iter=50, tol=1e-2):
    x = torch.zeros_like(b)

    r = b - apply_Gram(x, dx, dy, x_vec, y_vec, jx, jy, kx, ky)
    p = r.clone()

    rs_old = torch.dot(r, r)

    b_norm = torch.norm(b) + 1e-12  

    for i in range(max_iter):
        Ap = apply_Gram(p, dx, dy, x_vec, y_vec, jx, jy, kx, ky)

        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)

        x += alpha * p
        r -= alpha * Ap

        rs_new = torch.dot(r, r)

        rel_res = torch.sqrt(rs_new) / b_norm

        if rel_res < tol:
            print(f"CG converged in {i} iterations (rel_res={rel_res:.2e})")
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

