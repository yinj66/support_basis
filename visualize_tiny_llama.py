# qk_per_layer_prerope_llama_qk_one_figure.py
# Visualize pre-RoPE Q = X_l W_Q and K = X_l W_K entry distributions per layer (LLaMA/TinyLlama).
# - Correct for GQA: uses num_heads for Q and num_key_value_heads for K
# - Plots Q and K on the same figure per layer
# - Vertical dotted red lines at ±sqrt(log n_marker)
# - No V, no RoPE (we hook q_proj/k_proj outputs)
# - Works CPU/GPU; no accelerate required

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- Matplotlib global styling (8 pt everywhere) ----------------
plt.rcParams.update({
    "font.size": 8,          # base font size
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 8,
})

# ---------------- Config ----------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # swap to a Llama-2/3 checkpoint if you like
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else (
         torch.float16 if DEVICE == "cuda" else torch.float32)

BATCH_SIZE         = 2
ACTUAL_MAX_TOKENS  = 2048       # tokens actually processed (truncate/pad to this)
N_FOR_MARKER       = 2048       # n used only for drawing ±sqrt(log n) lines
USE_NATURAL_LOG    = True       # True: ln, False: log10
HEAD_INDEX         = 0          # which query head to inspect (maps to a KV head if GQA)
SUBSAMPLE          = 0          # e.g. 300_000 to speed plotting; 0 = no subsample
NUM_HIST_BINS      = 100

# synthetic long text to hit ACTUAL_MAX_TOKENS quickly
REPEAT_TOKEN = "h"
REPEAT_TIMES = 2048  # overkill; tokenizer will truncate

# ---------------- Load ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE if DEVICE == "cuda" else None
)
model.to(DEVICE).eval()
if DEVICE == "cuda":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ---------------- Helpers ----------------
def get_heads_numbers(attn, cfg):
    # robustly fetch heads for Q and KV (GQA)
    n_q = (getattr(attn, "num_heads", None) or
           getattr(attn, "num_attention_heads", None) or
           getattr(cfg, "num_attention_heads"))
    n_kv = (getattr(attn, "num_key_value_heads", None) or
            getattr(cfg, "num_key_value_heads", n_q))
    return int(n_q), int(n_kv)

def to_heads_q(btH, n_heads):
    # [B, T, Hq] -> [B, n_heads, T, d]
    B, T, H = btH.shape
    assert H % n_heads == 0, f"H={H} not divisible by n_heads={n_heads}"
    d = H // n_heads
    return btH.view(B, T, n_heads, d).transpose(1, 2).contiguous()

def to_heads_kv(btH, n_kv):
    # [B, T, Hkv] -> [B, n_kv, T, d]
    B, T, H = btH.shape
    assert H % n_kv == 0, f"H={H} not divisible by n_kv={n_kv}"
    d = H // n_kv
    return btH.view(B, T, n_kv, d).transpose(1, 2).contiguous()

def kv_index_for_query_head(q_head_idx, n_heads, n_kv):
    # map a query head index to its shared KV head (GQA grouping)
    group = max(1, n_heads // n_kv)
    return min(n_kv - 1, q_head_idx // group)

def flatten_cpu(x):  # [B, T, d] -> [N]
    return x.reshape(-1).float().cpu()

def maybe_subsample(vec, k):
    if k and vec.numel() > k:
        idx = torch.randint(0, vec.numel(), (k,))
        return vec[idx]
    return vec

def percentile_kth(x: torch.Tensor, p: float) -> float:
    n = x.numel()
    if n == 0: return float("nan")
    p = min(max(p, 0.0), 1.0)
    k = max(1, min(n, int(math.ceil(p * n))))
    if x.device.type != "cpu": x = x.cpu()
    x = x.view(-1)
    return x.kthvalue(k).values.item()

def vlines_with_labels(v, ax, txt, yfrac=0.9):
    ylim = ax.get_ylim()
    y = ylim[0] + yfrac * (ylim[1] - ylim[0])
    for sign in (+1, -1):
        x = sign * v
        ax.axvline(x, linestyle=":", linewidth=1.0, color="red")
        # ax.annotate(f"{txt} = {x:.3f}", xy=(x, y),
        #             xytext=(5 if sign>0 else -5, 0),
        #             textcoords="offset points", ha="left" if sign>0 else "right",
        #             va="center", color="red", fontsize=8)

def plot_qk_together(q_vec, k_vec, title, v_marker, filename, bins=NUM_HIST_BINS):
    # Build a common binning so the densities are directly comparable.
    if q_vec.numel() == 0 and k_vec.numel() == 0:
        print(f"Skip plot: {title} (both empty)")
        return
    q_np = q_vec.numpy() if q_vec.numel() else np.array([], dtype=np.float32)
    k_np = k_vec.numpy() if k_vec.numel() else np.array([], dtype=np.float32)

    all_vals = np.concatenate([q_np, k_np]) if q_np.size and k_np.size else (q_np if q_np.size else k_np)
    lo, hi = np.min(all_vals), np.max(all_vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1.0, 1.0  # fallback

    edges = np.linspace(lo, hi, bins)
    centers = 0.5 * (edges[1:] + edges[:-1])

    # --- Force figure size to 3.5 x 3.0 inches ---
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    if q_np.size:
        q_hist, _ = np.histogram(q_np, bins=edges, density=True)
        ax.plot(centers, q_hist, linewidth=1.0, label="Q")
    if k_np.size:
        k_hist, _ = np.histogram(k_np, bins=edges, density=True)
        ax.plot(centers, k_hist, linewidth=1.0, linestyle="--", label="K")

    if v_marker is not None and np.isfinite(v_marker):
        vlines_with_labels(v_marker, ax, r"$\sqrt{\log n}$")

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")

def print_stats(x, name):
    if x.numel() == 0:
        print(f"{name}: (no captures)")
        return
    mean = x.mean().item()
    std  = x.std(unbiased=False).item()
    p1   = percentile_kth(x, 0.01)
    p99  = percentile_kth(x, 0.99)
    print(f"{name}: n={x.numel():,}, mean={mean:.6f}, std={std:.6f}, p1={p1:.6f}, p99={p99:.6f}")

# ---------------- Prepare batch ----------------
long_text = (REPEAT_TOKEN * REPEAT_TIMES).strip()
TEXTS = [long_text] * BATCH_SIZE

enc = tokenizer(TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=ACTUAL_MAX_TOKENS)
input_ids = enc["input_ids"].to(DEVICE)
attention_mask = enc["attention_mask"].to(DEVICE)
seq_lens = attention_mask.sum(dim=1).tolist()

# n used for marker (purely visual)
n_marker = max(N_FOR_MARKER, 2)
v_marker = math.sqrt(math.log(n_marker)) if USE_NATURAL_LOG else math.sqrt(math.log10(n_marker))

# ---------------- Hooks: capture pre-RoPE Q/K ----------------
q_store = defaultdict(list)   # layer_idx -> [flattened vectors]
k_store = defaultdict(list)

def make_hook(layer_idx, which, attn_module, cfg):
    n_heads, n_kv = get_heads_numbers(attn_module, cfg)
    q_idx = min(HEAD_INDEX, n_heads - 1)
    kv_idx = kv_index_for_query_head(q_idx, n_heads, n_kv)

    def hook(_, __, out):
        t = out
        if t.dim() == 2:  # [T, H] -> [1, T, H]
            t = t.unsqueeze(0)
        # out is [B, T, H_proj]
        if which == "q":
            heads = to_heads_q(t, n_heads)           # [B, n_heads, T, d]
            one = heads[:, q_idx, :, :]              # [B, T, d]
            flat = flatten_cpu(maybe_subsample(one, SUBSAMPLE))
            q_store[layer_idx].append(flat)
        else:  # "k"
            heads = to_heads_kv(t, n_kv)             # [B, n_kv, T, d]
            one = heads[:, kv_idx, :, :]             # [B, T, d]
            flat = flatten_cpu(maybe_subsample(one, SUBSAMPLE))
            k_store[layer_idx].append(flat)
    return hook

# Register hooks
hooks = []
for i, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    hooks.append(attn.q_proj.register_forward_hook(make_hook(i, "q", attn, model.config)))
    hooks.append(attn.k_proj.register_forward_hook(make_hook(i, "k", attn, model.config)))

# Run a forward pass to trigger hooks
with torch.no_grad():
    _ = model(input_ids=input_ids, attention_mask=attention_mask)

# Remove hooks
for h in hooks:
    h.remove()

# ---------------- Meta & plots ----------------
n_tokens_max = int(max(seq_lens)) if seq_lens else ACTUAL_MAX_TOKENS
n_tokens_avg = float(np.mean(seq_lens)) if seq_lens else float(ACTUAL_MAX_TOKENS)

attn0 = model.model.layers[0].self_attn
n_heads0, n_kv0 = get_heads_numbers(attn0, model.config)
hidden = int(getattr(model.config, "hidden_size",
                     n_heads0 * (getattr(attn0, "head_dim", 0))))
head_dim = hidden // n_heads0 if n_heads0 else 0

print("\n=== Meta ===")
print(f"Processed tokens per example: max n = {n_tokens_max}, avg n ≈ {n_tokens_avg:.1f}")
print(f"Marker uses n = {n_marker}  ->  sqrt(log n) = {v_marker:.4f} "
      f"({'nat log' if USE_NATURAL_LOG else 'log10'})")
print(f"Hidden size H = {hidden}, num Q heads = {n_heads0}, num KV heads = {n_kv0}, head dim d = {head_dim}")
print(f"Analyzed query head index = {HEAD_INDEX}")

# For each layer, print stats and plot Q&K together
for layer_idx in sorted(set(list(q_store.keys()) + list(k_store.keys()))):
    q_vec = torch.cat(q_store[layer_idx], dim=0) if q_store[layer_idx] else torch.empty(0)
    k_vec = torch.cat(k_store[layer_idx], dim=0) if k_store[layer_idx] else torch.empty(0)

    print()  # spacer
    print_stats(q_vec, f"Layer {layer_idx:02d} — Q (pre-RoPE)")
    print_stats(k_vec, f"Layer {layer_idx:02d} — K (pre-RoPE, GQA-mapped)")

    plot_qk_together(
        q_vec, k_vec,
        title=f"Layer {layer_idx:02d} — Q & K Entry Distribution",
        v_marker=v_marker,
        filename=f"QK_{layer_idx}.pdf",
        bins=NUM_HIST_BINS
    )

print("\nDone. Figures saved as QK_<layer>.png in the current directory.")
