# pip install torch "transformers>=4.44,<5" pillow matplotlib

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, ViTModel

# =========================
# Config
# =========================
MODEL_ID = "google/vit-base-patch16-224-in21k"  # ViT-Base
DTYPE = torch.float32
device = torch.device("cpu")

TARGET_TOKENS = 2048              # desired n (CLS + patches)
TARGET_PATCHES = TARGET_TOKENS - 1
BATCH_SIZE = 1
OUT_DIR = "plots"
OUT_FILE = "vit_all_layers_overlaid.png"
DPI = 1200                        # high-res export
FIGSIZE = (16, 9)                 # larger canvas
GRID_POINTS = 512                 # KDE eval grid resolution
MAX_POINTS = 100_000              # cap samples per KDE to keep memory sane

# =========================
# Load model & processor
# =========================
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = ViTModel.from_pretrained(MODEL_ID)
model.to(device=device, dtype=DTYPE).eval()

# Patch size (ViT-Base uses 16)
patch = getattr(model.config, "patch_size", 16)

# =========================
# Helpers
# =========================
def best_factor_pair(n: int):
    """Return (h_patches, w_patches) with minimal aspect ratio for n = h*w."""
    best = None
    best_ratio = float("inf")
    r = int(math.sqrt(n))
    for a in range(1, r + 1):
        if n % a == 0:
            b = n // a
            a_, b_ = (a, b) if a <= b else (b, a)
            ratio = b_ / a_
            if ratio < best_ratio:
                best_ratio = ratio
                best = (a_, b_)
    return best  # (h, w)

def nearest_square_patches(n: int, prefer_ge=True):
    """Return k such that k*k is nearest square to n (ceil if prefer_ge)."""
    root = math.sqrt(n)
    k_lo, k_hi = math.floor(root), math.ceil(root)
    if prefer_ge:
        return k_hi
    return k_lo if (root - k_lo) <= (k_hi - root) else k_hi

def synth_images(k, H, W):
    """Make synthetic images exactly HxW."""
    imgs = []
    for _ in range(k):
        arr = np.random.rand(H, W, 3).astype(np.float32)
        gx = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
        gy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        arr = np.clip(0.6 * arr + 0.4 * (gx * gy), 0.0, 1.0)
        imgs.append(Image.fromarray((255 * arr).astype(np.uint8)))
    return imgs

def set_image_size_on_model(model, H, W):
    """Try to let ViT accept non-224 HxW by updating image_size everywhere."""
    try:
        model.config.image_size = (H, W)
    except Exception:
        try:
            model.config.image_size = max(H, W) if H == W else H
        except Exception:
            pass
    pe = getattr(getattr(getattr(model, "vit", None), "embeddings", None), "patch_embeddings", None)
    if pe is not None and hasattr(pe, "image_size"):
        try:
            pe.image_size = (H, W)
        except Exception:
            try:
                pe.image_size = max(H, W) if H == W else H
            except Exception:
                pass

def viT_forward(model, inputs):
    """Forward that tries interpolate_pos_encoding=True (newer HF) then falls back."""
    try:
        return model(**inputs, interpolate_pos_encoding=True)
    except TypeError:
        return model(**inputs)

# =========================
# Prepare target rectangular grid -> HxW
# =========================
h_rect, w_rect = best_factor_pair(TARGET_PATCHES)
H_rect, W_rect = h_rect * patch, w_rect * patch
print(f"Target rectangular patch grid {h_rect} x {w_rect} -> image {H_rect}x{W_rect} -> tokens n={1 + h_rect*w_rect}")

# Build images at that exact size
images_rect = synth_images(BATCH_SIZE, H_rect, W_rect)

# Disable resizing: keep exact HxW so we get CLS + h*w tokens
inputs_rect = processor(images=images_rect, return_tensors="pt", do_resize=False)
inputs_rect = {k: v.to(device) for k, v in inputs_rect.items()}

# Try to make the model accept HxW by updating its idea of image_size
set_image_size_on_model(model, H_rect, W_rect)

# Quick probe forward (without hooks) to see if this build supports rectangles
rect_ok = True
try:
    with torch.no_grad():
        _ = viT_forward(model, inputs_rect)
except ValueError as e:
    msg = str(e)
    if "Input image size" in msg and "doesn't match model" in msg:
        rect_ok = False
    else:
        raise

# If rectangle is rejected, fall back to nearest square >= target
if not rect_ok:
    k = nearest_square_patches(TARGET_PATCHES, prefer_ge=True)
    H_sq = W_sq = k * patch
    n_tokens_sq = 1 + k * k
    print(f"[fallback] Rectangular {H_rect}x{W_rect} rejected.")
    print(f"[fallback] Using square grid {k} x {k} -> image {H_sq}x{W_sq} -> tokens n={n_tokens_sq}")
    images_sq = synth_images(BATCH_SIZE, H_sq, W_sq)
    inputs_sq = processor(images=images_sq, return_tensors="pt", do_resize=False)
    inputs_sq = {k_: v_.to(device) for k_, v_ in inputs_sq.items()}
    set_image_size_on_model(model, H_sq, W_sq)
    inputs = inputs_sq
else:
    inputs = inputs_rect

# =========================
# Hook Q/K/V per layer
# =========================
q_per_layer, k_per_layer, v_per_layer = {}, {}, {}
Q_all, K_all, V_all = [], [], []
n_seq_global = {"n": None}

def _store(layer_idx, q=None, k=None, v=None):
    def vec(x): return x.detach().float().cpu().reshape(-1)
    if q is not None:
        qv = vec(q); Q_all.append(qv); q_per_layer.setdefault(layer_idx, []).append(qv)
        if n_seq_global["n"] is None: n_seq_global["n"] = q.shape[1]
    if k is not None:
        kv = vec(k); K_all.append(kv); k_per_layer.setdefault(layer_idx, []).append(kv)
        if n_seq_global["n"] is None: n_seq_global["n"] = k.shape[1]
    if v is not None:
        vv = vec(v); V_all.append(vv); v_per_layer.setdefault(layer_idx, []).append(vv)
        if n_seq_global["n"] is None: n_seq_global["n"] = v.shape[1]

def _layer_idx_from_name(name: str) -> int:
    parts = name.split(".")
    try:
        li = parts.index("layer")
        return int(parts[li + 1])
    except Exception:
        return -1

hooks = []
def make_hook_q(name):
    li = _layer_idx_from_name(name)
    def hook(module, inp, out): _store(li, q=out)  # [B, N, hidden]
    return hook
def make_hook_k(name):
    li = _layer_idx_from_name(name)
    def hook(module, inp, out): _store(li, k=out)
    return hook
def make_hook_v(name):
    li = _layer_idx_from_name(name)
    def hook(module, inp, out): _store(li, v=out)
    return hook

# ViT attention projections:
# vit.encoder.layer.{i}.attention.attention.{query,key,value}
for name, m in model.named_modules():
    lname = name.lower()
    if lname.endswith("attention.attention.query") or lname.endswith("attention.self.query") or lname.endswith(".query"):
        hooks.append(m.register_forward_hook(make_hook_q(name)))
    elif lname.endswith("attention.attention.key") or lname.endswith("attention.self.key") or lname.endswith(".key"):
        hooks.append(m.register_forward_hook(make_hook_k(name)))
    elif lname.endswith("attention.attention.value") or lname.endswith("attention.self.value") or lname.endswith(".value"):
        hooks.append(m.register_forward_hook(make_hook_v(name)))

# Forward (now with hooks)
with torch.no_grad():
    _ = viT_forward(model, inputs)

for h in hooks: h.remove()

def cat_or_empty(lst): return torch.cat(lst) if lst else torch.empty(0)
Q_all = cat_or_empty(Q_all); K_all = cat_or_empty(K_all); V_all = cat_or_empty(V_all)
Q_layer = {i: cat_or_empty(vs) for i, vs in q_per_layer.items()}
K_layer = {i: cat_or_empty(vs) for i, vs in k_per_layer.items()}
V_layer = {i: cat_or_empty(vs) for i, vs in v_per_layer.items()}

# =========================
# Robust percentiles
# =========================
def percentile_kth(x: torch.Tensor, p: float) -> float:
    n = x.numel()
    if n == 0: return float("nan")
    p = min(max(p, 0.0), 1.0)
    k = max(1, min(n, int(math.ceil(p * n))))
    x = x.view(-1).cpu()
    return x.kthvalue(k).values.item()

# n and threshold
n_seq = int(n_seq_global["n"]) if n_seq_global["n"] is not None else 0
thr = math.sqrt(math.log(n_seq)) if n_seq > 1 else 0.0

# =========================
# KDE utilities (same as for GPT script)
# =========================
def _silverman_bandwidth(x: torch.Tensor) -> float:
    n = x.numel()
    if n <= 1: return 1.0
    std = x.std(unbiased=False).item()
    return 1.06 * std * (n ** (-1/5)) if std > 0 and math.isfinite(std) else 1.0

def _kde_gaussian(x: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    x = x.view(1, -1)         # (1, N)
    g = grid.view(-1, 1)      # (G, 1)
    u = (g - x) / h
    K = torch.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)
    return K.mean(dim=1) / h  # (G,)

def _make_global_grid(tensors, num=GRID_POINTS, pad=0.1) -> torch.Tensor:
    vals = [t.detach().float().cpu().view(-1) for t in tensors if t.numel() > 0]
    if not vals:
        return torch.linspace(-1, 1, steps=num)
    x = torch.cat(vals)
    p1, p99 = percentile_kth(x, 0.01), percentile_kth(x, 0.99)
    lo, hi = (p1, p99) if math.isfinite(p1) and math.isfinite(p99) else (x.min().item(), x.max().item())
    if lo == hi: lo, hi = lo - 1.0, hi + 1.0
    span = hi - lo
    lo -= pad * span
    hi += pad * span
    return torch.linspace(lo, hi, steps=num, dtype=torch.float32)

def kde_curve(vec: torch.Tensor, grid: torch.Tensor, max_points=MAX_POINTS) -> torch.Tensor | None:
    if vec.numel() == 0: return None
    x = vec.detach().float().cpu().view(-1)
    if x.numel() > max_points:
        idx = torch.randperm(x.numel())[:max_points]
        x = x[idx]
    h = _silverman_bandwidth(x)
    return _kde_gaussian(x, grid, h)

# =========================
# Collect datasets (aggregate + per-layer)
# =========================
datasets = [
    ("Q (all)", Q_all),
    ("K (all)", K_all),
    ("V (all)", V_all),
]
for i in sorted(set(Q_layer) | set(K_layer) | set(V_layer)):
    if i < 0: continue
    datasets.append((f"L{i} Q", Q_layer.get(i, torch.empty(0))))
    datasets.append((f"L{i} K", K_layer.get(i, torch.empty(0))))
    datasets.append((f"L{i} V", V_layer.get(i, torch.empty(0))))

# =========================
# Plot ALL curves in one Cartesian axis
# =========================
os.makedirs(OUT_DIR, exist_ok=True)
grid = _make_global_grid([vec for _, vec in datasets], num=GRID_POINTS)

plt.figure(figsize=FIGSIZE)
for label, vec in datasets:
    dens = kde_curve(vec, grid)
    if dens is not None:
        plt.plot(grid.numpy(), dens.numpy(), label=label, linewidth=0.9)

# Reference lines ±√ln n
plt.axvline(+thr, linestyle=":", color="red")
plt.axvline(-thr, linestyle=":", color="red")

plt.title(f"ViT Q/K/V (KDE); n={n_seq}, ±√ln n≈{thr:.3f}", fontsize=16)
plt.xlabel("value", fontsize=13)
plt.ylabel("density", fontsize=13)

# Big legend outside the plot (right side), many columns for compactness
leg = plt.legend(
    fontsize=9, ncol=6, loc="center left",
    bbox_to_anchor=(1.02, 0.5), frameon=True, title="Series"
)
if leg.get_title():
    leg.get_title().set_fontsize(10)

plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave room for the outside legend

out_path = os.path.join(OUT_DIR, OUT_FILE)
plt.savefig(out_path, dpi=DPI)
plt.close()
print(f"[saved] {out_path}")
