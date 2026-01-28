# visualize_llada_kde.py
# Empirical PDFs (KDE) of Q = X_l W_Q and K = X_l W_K for LLaDA-8B.
# Single-axis plot with all per-layer curves, plus optional V and "all-layers" curves.
# Draws red dotted lines at ±sqrt(log n), where n is the number of input tokens.

import argparse
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from transformers import AutoModel, AutoTokenizer


# ---------------------------
# Helpers
# ---------------------------

def to_cpu_flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", non_blocking=True).float().reshape(-1)

def layer_name(i: int) -> str:
    return f"L{i}"

def get_blocks(core_transformer) -> List[nn.Module]:
    if hasattr(core_transformer, "blocks"):
        return list(core_transformer.blocks)
    if hasattr(core_transformer, "block_groups"):
        return [b for bg in core_transformer.block_groups for b in bg]
    raise RuntimeError("Could not find transformer.blocks or transformer.block_groups")

def kde_curve(x: np.ndarray, grid: np.ndarray, bw: str | float | None = None) -> np.ndarray:
    """
    Return KDE(x) evaluated on `grid`.
    bw: None/'scott'/'silverman' or a float scaling factor.
    """
    if x.size == 0:
        return np.zeros_like(grid)
    try:
        kde = gaussian_kde(x, bw_method=bw)
        return kde.evaluate(grid)
    except Exception:
        # Fall back to a slightly broadened Scott bandwidth if numerical issues occur
        kde = gaussian_kde(x, bw_method="scott")
        kde.set_bandwidth(kde.factor * 1.5)
        return kde.evaluate(grid)

def make_grid(arrays: List[np.ndarray], pad_std: float = 4.0, npts: int = 4096) -> np.ndarray:
    """Build a symmetric grid around 0 large enough for all arrays."""
    allx = np.concatenate([a for a in arrays if a.size > 0]) if arrays else np.array([0.0])
    m = float(np.mean(allx))
    s = float(np.std(allx)) + 1e-8
    half = max(abs(allx.min() - m), abs(allx.max() - m), pad_std * s)
    # Make it symmetric around 0 to align Q/K shapes visually
    half = max(half, 1.0)
    return np.linspace(-half, half, npts)

# ---------------------------
# Hook registration (Q/K/+V)
# ---------------------------

def register_qkv_hooks_llada(model, store, include_v: bool):
    """
    Collects flattened entries of Q, K (and optionally V) per layer.
    Handles both separate q/k(/v) and fused att_proj + fused_dims layouts.
    """
    hooks = []
    core = model.model                     # LLaDA keeps the core under .model
    tr = core.transformer
    blocks = get_blocks(tr)

    store.setdefault("per_layer", {})      # per_layer[lname] = {"Q":[...], "K":[...], "V":[...optional]}
    for li, block in enumerate(blocks):
        lname = layer_name(li)
        store["per_layer"].setdefault(lname, {"Q": [], "K": []})
        if include_v:
            store["per_layer"][lname].setdefault("V", [])

        # Separate projections
        if hasattr(block, "q_proj") and hasattr(block, "k_proj"):
            def mk_q(lname=lname):
                def hook(_, __, out):
                    store["per_layer"][lname]["Q"].append(to_cpu_flat(out))
                return hook
            def mk_k(lname=lname):
                def hook(_, __, out):
                    store["per_layer"][lname]["K"].append(to_cpu_flat(out))
                return hook
            hooks.append(block.q_proj.register_forward_hook(mk_q()))
            hooks.append(block.k_proj.register_forward_hook(mk_k()))

            if include_v and hasattr(block, "v_proj"):
                def mk_v(lname=lname):
                    def hook(_, __, out):
                        store["per_layer"][lname]["V"].append(to_cpu_flat(out))
                    return hook
                hooks.append(block.v_proj.register_forward_hook(mk_v()))
            continue

        # Fused QKV
        if hasattr(block, "att_proj") and hasattr(block, "fused_dims"):
            q_dim, k_dim, v_dim = block.fused_dims
            def mk_att(lname=lname, q_dim=q_dim, k_dim=k_dim, v_dim=v_dim):
                def hook(_, __, out):
                    q, k, v = out.split((q_dim, k_dim, v_dim), dim=-1)
                    store["per_layer"][lname]["Q"].append(to_cpu_flat(q))
                    store["per_layer"][lname]["K"].append(to_cpu_flat(k))
                    if include_v:
                        store["per_layer"][lname]["V"].append(to_cpu_flat(v))
                return hook
            hooks.append(block.att_proj.register_forward_hook(mk_att()))
            continue

        raise RuntimeError(f"{lname}: expected (q_proj,k_proj) or (att_proj,fused_dims)")

    return hooks

# ---------------------------
# Plot (single axes with all curves)
# ---------------------------

def plot_all_kde(store: Dict[str, Any], out_png: str, tok_count: int, logy: bool, bw: str | float | None):
    # flatten per-layer tensors -> numpy arrays
    per_layer_np: Dict[str, Dict[str, np.ndarray]] = {}
    have_V = False
    for lname, d in store["per_layer"].items():
        per_layer_np[lname] = {}
        for k in ("Q", "K", "V"):
            if k in d and len(d[k]):
                arr = torch.cat(d[k]).numpy()
                per_layer_np[lname][k] = arr
                if k == "V":
                    have_V = True

    # Build a common grid large enough for everything
    arrays_for_grid = []
    for d in per_layer_np.values():
        arrays_for_grid += list(d.values())
    grid = make_grid(arrays_for_grid, pad_std=6.0, npts=4096)

    # Prepare overall "all layers" arrays
    allQ = np.concatenate([d["Q"] for d in per_layer_np.values() if "Q" in d]) if any("Q" in d for d in per_layer_np.values()) else np.array([])
    allK = np.concatenate([d["K"] for d in per_layer_np.values() if "K" in d]) if any("K" in d for d in per_layer_np.values()) else np.array([])
    allV = np.concatenate([d["V"] for d in per_layer_np.values() if "V" in d]) if have_V else np.array([])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Overall curves first (thicker)
    if allQ.size:
        ax.plot(grid, kde_curve(allQ, grid, bw), label="Q (all)", linewidth=2.0)
    if allK.size:
        ax.plot(grid, kde_curve(allK, grid, bw), label="K (all)", linewidth=2.0)
    if allV.size:
        ax.plot(grid, kde_curve(allV, grid, bw), label="V (all)", linewidth=2.0)

    # Per-layer curves
    for lname in sorted(per_layer_np.keys(), key=lambda s: int(s[1:])):  # sort by layer index
        d = per_layer_np[lname]
        if "Q" in d:
            ax.plot(grid, kde_curve(d["Q"], grid, bw), alpha=0.9, linewidth=1.0, label=f"{lname} Q")
        if "K" in d:
            ax.plot(grid, kde_curve(d["K"], grid, bw), alpha=0.9, linewidth=1.0, label=f"{lname} K")
        if "V" in d:
            ax.plot(grid, kde_curve(d["V"], grid, bw), alpha=0.9, linewidth=1.0, label=f"{lname} V")

    # ±sqrt(log n) guides
    import math
    thr = math.sqrt(math.log(max(2, tok_count)))  # safe for n>=2
    for s in (-thr, +thr):
        ax.axvline(s, linestyle=":", color="red", linewidth=1.5)

    ax.set_title(f"Empirical PDFs of Q/K{'/V' if have_V else ''} (KDE); n={tok_count},  ±√ln n≈{thr:.3f}")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    if logy:
        ax.set_yscale("log")
    ax.legend(ncol=5, fontsize=8, frameon=True)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--revision", default=None, help="Pin HF revision/commit (optional)")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32","bfloat16","float16"])
    ap.add_argument("--out", default="qkv_kde_llada.png", help="Output PNG path")
    ap.add_argument("--logy", action="store_true", help="Log-scale y-axis")
    ap.add_argument("--include_v", action="store_true", help="Also plot V")
    ap.add_argument("--bw", default=None,
                    help="KDE bandwidth: None/'scott'/'silverman' or float scale (e.g., 0.6)")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, revision=args.revision)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        revision=args.revision,
        dtype=dtype,                # (torch_dtype is deprecated)
    ).to(args.device)
    model.eval()

    # Capture Q/K(/V)
    store: Dict[str, Any] = {}
    hooks = register_qkv_hooks_llada(model, store, include_v=args.include_v)

    # Forward pass
    with torch.no_grad():
        inputs = tok(args.prompt, return_tensors="pt").to(args.device)
        seq_len = int(inputs["input_ids"].shape[1])  # n = number of tokens
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    # One combined figure with KDE curves + ±sqrt(log n)
    plot_all_kde(store, args.out, tok_count=seq_len, logy=args.logy,
                 bw=(float(args.bw) if (args.bw and args.bw.replace('.','',1).isdigit()) else (args.bw if args.bw in (None, "scott", "silverman") else None)))

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
