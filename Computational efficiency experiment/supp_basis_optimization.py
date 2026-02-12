import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from numpy.polynomial.chebyshev import chebfit, cheb2poly

# =======================
# Config
# =======================
N = 4096
d = 64
DEGREE = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

torch.backends.cuda.matmul.allow_tf32 = False
if device == "cuda":
    torch.set_float32_matmul_precision("highest")

LOCK_TIGHT_POLY = True       # keep small-block polynomial interval tight when T<=0.1
SANITY_BOUND_CHECK = False
SAMPLE_FOR_CHECK = 2048
KERNEL_CHUNK = 2048          # chunk size for building S or A in blocks to bound memory

# =======================
# Timing
# =======================
def walltime(fn, repeats=1):
    best = 1e9; out = None
    for _ in range(repeats):
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device == "cuda": torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, out

# =======================
# Chebyshev -> monomial (cached; not timed)
# =======================
def cheb_coeff_exp(a, b, degree):
    xs = np.cos((2*np.arange(1, degree+2)-1)/(2*(degree+1))*np.pi)
    xs = 0.5*(b-a)*xs + 0.5*(b+a)
    ys = np.exp(xs)
    return chebfit(xs, ys, degree)

def poly_coeff_from_cheb(a, b, degree):
    c = cheb_coeff_exp(a, b, degree)
    p = cheb2poly(c)
    if len(p) < degree+1:
        p = np.pad(p, (0, degree+1-len(p)))
    return p

@lru_cache(maxsize=None)
def get_mono_coeffs_cached(B: float, degree: int):
    B = float(max(B, 1e-12))
    a = poly_coeff_from_cheb(-B, B, degree)
    return tuple(float(x) for x in a[:degree+1])

def split_coeff(a):
    if a == 0.0: return 0.0, 0.0
    s = 1.0 if a > 0 else -1.0
    r = math.sqrt(abs(a))
    return r, s*r

# =======================
# Feature blocks
# =======================
def features_deg2(X):
    n, D = X.shape
    sq = X**2
    idx = [(i,j) for i in range(D) for j in range(i+1, D)]
    if not idx:
        cross = torch.empty(n, 0, device=X.device, dtype=X.dtype)
    else:
        I = torch.tensor([i for i,j in idx], device=X.device)
        J = torch.tensor([j for i,j in idx], device=X.device)
        cross = math.sqrt(2.0) * (X[:,I] * X[:,J])
    return sq, cross

def features_deg3(X):
    n, D = X.shape
    cu = X**3
    idx_ij = [(i,j) for i in range(D) for j in range(D) if i!=j]
    if not idx_ij:
        sqx = torch.empty(n, 0, device=X.device, dtype=X.dtype)
    else:
        I = torch.tensor([i for i,j in idx_ij], device=X.device)
        J = torch.tensor([j for i,j in idx_ij], device=X.device)
        sqx = math.sqrt(3.0) * ((X[:,I]**2) * X[:,J])
    idx_ijk = [(i,j,k) for i in range(D) for j in range(i+1,D) for k in range(j+1,D)]
    if not idx_ijk:
        tri = torch.empty(n, 0, device=X.device, dtype=X.dtype)
    else:
        I = torch.tensor([i for i,j,k in idx_ijk], device=X.device)
        J = torch.tensor([j for i,j,k in idx_ijk], device=X.device)
        K = torch.tensor([k for i,j,k in idx_ijk], device=X.device)
        tri = math.sqrt(6.0) * (X[:,I] * X[:,J] * X[:,K])
    return cu, sqx, tri

# =======================
# Polynomial factors (deg ≤3)
# =======================
def apply_poly_rank3(Q, K, a0, a1, a2, a3):
    n, D = Q.shape
    dev, dt = Q.device, Q.dtype
    Ls = []; Rs = []
    l0, r0 = split_coeff(float(a0))
    Ls.append(torch.full((n,1), l0, device=dev, dtype=dt))
    Rs.append(torch.full((n,1), r0, device=dev, dtype=dt))
    if a1 != 0.0:
        l1, r1 = split_coeff(float(a1))
        Ls.append(l1 * Q); Rs.append(r1 * K)
    if a2 != 0.0:
        l2, r2 = split_coeff(float(a2))
        Q_sq, Q_cr = features_deg2(Q); K_sq, K_cr = features_deg2(K)
        Ls.append(l2 * Q_sq); Rs.append(r2 * K_sq)
        if Q_cr.shape[1] > 0:
            Ls.append(l2 * Q_cr); Rs.append(r2 * K_cr)
    if a3 != 0.0:
        l3, r3 = split_coeff(float(a3))
        Q_cu, Q_sqx, Q_tri = features_deg3(Q); K_cu, K_sqx, K_tri = features_deg3(K)
        Ls.append(l3 * Q_cu);  Rs.append(r3 * K_cu)
        if Q_sqx.shape[1] > 0:
            Ls.append(l3 * Q_sqx); Rs.append(r3 * K_sqx)
        if Q_tri.shape[1] > 0:
            Ls.append(l3 * Q_tri); Rs.append(r3 * K_tri)
    L = torch.cat(Ls, dim=1)
    R = torch.cat(Rs, dim=1)
    return L, R

def apply_AS23_factors_from_coeffs(Q, K, coeffs_tuple):
    a0 = coeffs_tuple[0]
    a1 = coeffs_tuple[1] if len(coeffs_tuple) >= 2 else 0.0
    a2 = coeffs_tuple[2] if len(coeffs_tuple) >= 3 else 0.0
    a3 = coeffs_tuple[3] if len(coeffs_tuple) >= 4 else 0.0
    L, R = apply_poly_rank3(Q, K, a0, a1, a2, a3)
    return (L, R), a0

# =======================
# Attention & kernel helpers
# =======================
def exact_attention(Q,K,V, scale=1.0):
    A = torch.exp((Q @ K.T) * scale)
    Dinv = 1.0 / (A.sum(dim=1, keepdim=True))
    return (A @ V) * Dinv

def as23_attention(Q,K,V, scale=1.0, coeffs=None):
    assert coeffs is not None
    srt = math.sqrt(abs(scale))
    (L, R), _ = apply_AS23_factors_from_coeffs(Q*srt, K*srt, coeffs)
    ones = torch.ones(R.size(0),1, device=Q.device, dtype=Q.dtype)
    d1 = L @ (R.T @ ones)
    C1 = L @ (R.T @ V)
    return C1 * (1.0 / d1.clamp_min(1e-30))

@torch.no_grad()
def big_small_attention(Q,K,V, T, scale=1.0, coeffs_small=None, chunk=4096):
    assert coeffs_small is not None
    n, D = Q.shape
    dev, dt = Q.device, Q.dtype
    R_mask = (Q.abs() > T).any(dim=1)
    C_mask = (K.abs() > T).any(dim=1)
    R_idx = torch.where(R_mask)[0]
    C_idx = torch.where(C_mask)[0]
    notR_idx = torch.where(~R_mask)[0]
    Qs = Q.clone(); Qs[R_mask] = 0
    Ks = K.clone(); Ks[C_mask] = 0
    srt = math.sqrt(abs(scale))
    (L_s, R_s), _ = apply_AS23_factors_from_coeffs(Qs*srt, Ks*srt, coeffs_small)
    ones = torch.ones(n,1, device=dev, dtype=dt)
    d1 = L_s @ (R_s.T @ ones)
    C1 = L_s @ (R_s.T @ V)
    d2 = torch.zeros(n,1, device=dev, dtype=dt)
    C2 = torch.zeros(n,D, device=dev, dtype=dt)
    KT = K.T.contiguous()
    expm1_ = torch.expm1
    for s0 in range(0, len(R_idx), chunk):
        rr = R_idx[s0:s0+chunk]
        S_R = expm1_((Q[rr] @ KT) * scale)
        d2[rr,0] = S_R.sum(dim=1)
        C2[rr]   = S_R @ V
    if C_idx.numel() > 0:
        KC = K[C_idx]; KCT = KC.T.contiguous()
        VC = V[C_idx]
        for s0 in range(0, len(notR_idx), chunk):
            rr = notR_idx[s0:s0+chunk]
            S_C = expm1_((Q[rr] @ KCT) * scale)
            d2[rr,0] += S_C.sum(dim=1)
            C2[rr]   += S_C @ VC
    Dinv = 1.0 / (d1 + d2).clamp_min(1e-30)
    Y = (C1 + C2) * Dinv
    stats = {
        "num_rows_big": int(R_mask.sum()),
        "num_cols_big": int(C_mask.sum()),
        "d1_sum": float(d1.sum().item()),
        "d2_sum": float(d2.sum().item()),
    }
    return Y, stats

# ----- kernel builders (chunked) -----
@torch.no_grad()
def kernel_exp(Q, K, scale=1.0, chunk=KERNEL_CHUNK):
    n = Q.shape[0]
    out = torch.empty(n, n, device=Q.device, dtype=Q.dtype)
    KT = K.T.contiguous()
    for s in range(0, n, chunk):
        rr = slice(s, min(n, s+chunk))
        out[rr] = torch.exp((Q[rr] @ KT) * scale)
    return out

@torch.no_grad()
def kernel_as23(Q, K, scale, coeffs, chunk=KERNEL_CHUNK):
    # Build A_hat = L R^T without forming full L,R in memory
    srt = math.sqrt(abs(scale))
    (L, R), _ = apply_AS23_factors_from_coeffs(Q*srt, K*srt, coeffs)
    return L @ R.T

@torch.no_grad()
def kernel_big_small(Q, K, T, scale, coeffs_small, chunk=KERNEL_CHUNK):
    n = Q.shape[0]
    R_mask = (Q.abs() > T).any(dim=1)
    C_mask = (K.abs() > T).any(dim=1)
    Qs = Q.clone(); Qs[R_mask] = 0
    Ks = K.clone(); Ks[C_mask] = 0
    # Polynomial on small block
    srt = math.sqrt(abs(scale))
    (L_s, R_s), _ = apply_AS23_factors_from_coeffs(Qs*srt, Ks*srt, coeffs_small)
    A_small = L_s @ R_s.T  # approx exp on small-small
    # Exact correction on large part: exp(A_L) - 1
    A = A_small.clone()
    KT = K.T.contiguous()
    # rows in R: all cols
    rows = torch.where(R_mask)[0]
    if rows.numel() > 0:
        S_R = torch.exp((Q[rows] @ KT) * scale) - 1.0
        A[rows] += S_R
    # other rows: only big columns C
    cols = torch.where(C_mask)[0]
    if cols.numel() > 0:
        notR = torch.where(~R_mask)[0]
        if notR.numel() > 0:
            KCT = K[cols].T.contiguous()
            S_C = torch.exp((Q[notR] @ KCT) * scale) - 1.0
            A[notR[:,None], cols[None,:]] += S_C
    return A

def rel_frob(A, B):
    return (A - B).norm() / B.norm()

# =======================
# Driver
# =======================
if __name__ == "__main__":
    torch.manual_seed(123)
    Q = 0.1*torch.randn(N,d, device=device, dtype=dtype)
    K = 0.1*torch.randn(N,d, device=device, dtype=dtype)
    V = 0.05*torch.randn(N,d, device=device, dtype=dtype)
    scale = 1.0/float(d)

    # Exact baselines (attention outputs)
    t_exact, Y_exact = walltime(lambda: exact_attention(Q,K,V, scale), repeats=1)

    # AS23 polynomials (fixed intervals)
    qmax = Q.norm(dim=1).max().item();  kmax = K.norm(dim=1).max().item()
    B_AS23 = float(abs(scale) * qmax * kmax)
    coeffs_AS23 = get_mono_coeffs_cached(B_AS23, DEGREE)

    # Baseline errors (attention outputs)
    _, Y_as23_once = walltime(lambda: as23_attention(Q,K,V, scale, coeffs=coeffs_AS23), repeats=1)
    base_as23_err = (Y_as23_once - Y_exact).norm() / Y_exact.norm()

    # Threshold sweep
    Ts = torch.linspace(0.15, 0.5, steps=30, device=device).tolist()
    th_x = []
    # attention-output errors
    err_as23 = []; t_as23 = []
    err_bs = [];   t_bs = []
    # kernel errors
    kerr_as23 = []; kerr_bs = []; kerr_clip = []

    # Pre-build full kernels once for denominator norms (to avoid repeating work)
    A_exact = kernel_exp(Q, K, scale)
    A_as23_full = kernel_as23(Q, K, scale, coeffs_AS23)
    kden_exact = A_exact.norm()

    for T in Ts[1:]:
        th_x.append(float(T))

        # --- attention-output errors
        tA, Y_as23_cur = walltime(lambda: as23_attention(Q,K,V, scale, coeffs=coeffs_AS23), repeats=1)
        t_as23.append(float(tA))
        err_as23.append(float((Y_as23_cur - Y_exact).norm() / Y_exact.norm()))

        B_small = float(T*T)
        if LOCK_TIGHT_POLY: B_small = float(min(B_small, 0.1*0.1))
        coeffs_small_T = get_mono_coeffs_cached(B_small, DEGREE)

        tB, (Y_bs, _) = walltime(lambda: big_small_attention(Q,K,V, T=T, scale=scale, coeffs_small=coeffs_small_T), repeats=1)
        t_bs.append(float(tB))
        err_bs.append(float((Y_bs - Y_exact).norm() / Y_exact.norm()))

        # --- kernel errors (Frobenius-relative)
        A_bs = kernel_big_small(Q, K, T, scale, coeffs_small_T)
        kerr_bs.append(float((A_bs - A_exact).norm() / kden_exact))

        # AS23 fixed (original)
        kerr_as23.append(float((A_as23_full - A_exact).norm() / kden_exact))

    # ======= Plots: attention outputs =======
    plt.figure()
    plt.plot(th_x, err_as23, marker='o', label=f'AS23 (deg={DEGREE}) vs original exact')
    plt.plot(th_x, err_bs,   marker='o', label='Single-threshold (ours) vs original exact')
    plt.xlabel("Threshold T"); plt.ylabel("Rel. error (attention output)")
    plt.title("Attention Output Error vs Threshold")
    plt.legend(); plt.grid(True)

    # ======= Plots: kernel errors (this shows your expected trend) =======
    plt.figure()
    plt.plot(th_x, kerr_as23, marker='o', label='AS23 vs exact attention matrix')
    plt.plot(th_x, kerr_bs,   marker='o', label='Single-threshold (ours) vs exact attention matrix')
    print("Threshold: " + str(th_x))
    print("AS23 error: " + str(kerr_as23))
    print("Our error: " + str(kerr_bs))
    print("AS23 time: " + str(t_as23))
    print("Our time: " + str(t_bs))
    print(f"Exact time: {t_exact:.6f} s")
    plt.xlabel("Threshold T"); plt.ylabel("Rel. Frobenius error (kernel)")
    plt.title("Kernel Approximation Error vs Threshold")
    plt.legend(); plt.grid(True)

    # ======= Runtime plot (optional) =======
    plt.figure()
    plt.plot(th_x, t_as23, marker='o', label=f'AS23 (deg={DEGREE})')
    plt.plot(th_x, t_bs,   marker='o', label='Single-threshold (ours)')
    plt.axhline(walltime(lambda: exact_attention(Q,K,V, scale), repeats=1)[0],
                linestyle='--', label='Exact (single run)')
    plt.xlabel("Threshold T"); plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Threshold")
    plt.legend(); plt.grid(True)
    plt.show()

