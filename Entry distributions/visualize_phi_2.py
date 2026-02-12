# visualize_qk_phi2.py
# pip install torch transformers accelerate scipy matplotlib

import math
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "microsoft/phi-2"

@dataclass
class Hooks:
    q: Dict[int, torch.Tensor]
    k: Dict[int, torch.Tensor]

def _find_qk_modules(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module, Optional[torch.nn.Module]]]:
    """
    Return [(layer_idx, q_module, k_module)] for all attention layers.
    Tries common names used across HF models. For fused QKV, returns (Wqkv, None)
    and we'll split it later.
    """
    hits = []
    for idx, m in enumerate(model.modules()):
        name = type(m).__name__.lower()
        # look for attention blocks that contain q/k linears
        if "attention" in name or "attn" in name:
            # try to extract children
            q_mod = getattr(m, "q_proj", None) or getattr(m, "query", None)
            k_mod = getattr(m, "k_proj", None) or getattr(m, "key", None)
            qkv_mod = getattr(m, "qkv", None) or getattr(m, "Wqkv", None) or getattr(m, "wqkv", None) or getattr(m, "c_attn", None)
            if q_mod is not None and k_mod is not None:
                hits.append((len(hits), q_mod, k_mod))
            elif qkv_mod is not None:
                hits.append((len(hits), qkv_mod, None))
    # Fallback: traverse known phi-style blocks
    if not hits:
        for lid, block in enumerate(getattr(model, "transformer", getattr(model, "model", model)).modules()):
            if hasattr(block, "q_proj") or hasattr(block, "k_proj") or hasattr(block, "qkv"):
                q = getattr(block, "q_proj", None)
                k = getattr(block, "k_proj", None)
                qkv = getattr(block, "qkv", None)
                if q is not None and k is not None:
                    hits.append((len(hits), q, k))
                elif qkv is not None:
                    hits.append((len(hits), qkv, None))
    return hits

def _register_hooks(model: torch.nn.Module, device: torch.device) -> Hooks:
    """
    Attach forward hooks to capture Q and K per layer. Handles fused QKV if present.
    """
    hooks = Hooks(q={}, k={})
    handles = []

    def make_store(side: str, layer_idx: int):
        def store(_, __, output):
            # output: (batch, seq, dim) or (seq, batch, dim) – normalize to (B,S,D)
            t = output
            if t.dim() == 3 and t.shape[0] > 8 and t.shape[0] > t.shape[1]:  # likely (S,B,D)
                t = t.transpose(0, 1)
            hooks.__dict__[side][layer_idx] = t.detach().to("cpu")
        return store

    # For fused QKV, we’ll split assuming last dim = 3*D
    def make_store_qkv(layer_idx: int):
        def store(_, __, output):
            t = output
            if t.dim() == 3 and t.shape[0] > 8 and t.shape[0] > t.shape[1]:  # (S,B,D)
                t = t.transpose(0, 1)
            B, S, D3 = t.shape
            assert D3 % 3 == 0, "Expected fused QKV last dim to be 3*D"
            D = D3 // 3
            q = t[..., :D]
            k = t[..., D:2*D]
            hooks.q[layer_idx] = q.detach().to("cpu")
            hooks.k[layer_idx] = k.detach().to("cpu")
        return store

    for layer_idx, q_mod, k_mod in _find_qk_modules(model):
        if k_mod is None:  # fused
            handles.append(q_mod.register_forward_hook(make_store_qkv(layer_idx)))
        else:
            handles.append(q_mod.register_forward_hook(make_store("q", layer_idx)))
            handles.append(k_mod.register_forward_hook(make_store("k", layer_idx)))

    # Save a remover so we can clean up later
    hooks._remove = lambda: [h.remove() for h in handles]  # type: ignore[attr-defined]
    return hooks

def kde_pdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(x.astype(np.float64))
    return kde(grid)

def flatten_entries(t: torch.Tensor) -> np.ndarray:
    return t.reshape(-1).float().numpy()

def plot_kdes(
    series: List[Tuple[str, np.ndarray]],
    n_tokens: int,
    title_prefix: str = "Empirical PDFs of Q/K (KDE)"
):
    # Common x-grid across all series, based on pooled quantiles for stability
    pooled = np.concatenate([s for _, s in series])
    q_lo, q_hi = np.quantile(pooled, [0.001, 0.999])
    span = max(abs(q_lo), abs(q_hi))
    x = np.linspace(-max(3.0, 1.2*span), max(3.0, 1.2*span), 2000)

    plt.figure(figsize=(14, 8))
    for label, data in series:
        y = kde_pdf(data, x)
        plt.plot(x, y, linewidth=1.0, label=label)

    # red dotted verticals at ±sqrt(ln n)
    thr = math.sqrt(math.log(max(2, n_tokens)))
    for sgn in (-1, 1):
        plt.axvline(sgn * thr, color="red", linestyle=":", linewidth=1.5)

    plt.title(f"{title_prefix}; n={n_tokens}, ±√ln n≈{thr:.3f}")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.legend(ncol=6, fontsize=8)
    plt.tight_layout()
    plt.show()

def run_with_phi2(prompt: str, target_seq_len: int, device: str = "auto"):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Load tokenizer & model
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=True
        )
        model.eval()

        # Build input close to desired n
        ids = tok(prompt, return_tensors="pt").input_ids
        # Autoregressively extend to ~target_seq_len (using tokenizer's pad/eos as needed)
        if ids.shape[-1] < target_seq_len:
            pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            pad = torch.full((1, target_seq_len - ids.shape[-1]), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=-1)
        elif ids.shape[-1] > target_seq_len:
            ids = ids[:, :target_seq_len]

        device0 = next(model.parameters()).device
        ids = ids.to(device0)

        # Register hooks & forward
        hooks = _register_hooks(model, device0)
        with torch.no_grad():
            _ = model(input_ids=ids)

        # Collect flattened entries across layers
        series = []
        # "All layers combined"
        if hooks.q:
            all_q = np.concatenate([flatten_entries(t) for t in hooks.q.values()])
            series.append(("Q (all)", all_q))
        if hooks.k:
            all_k = np.concatenate([flatten_entries(t) for t in hooks.k.values()])
            series.append(("K (all)", all_k))

        # Per-layer entries (limit legend clutter if many)
        for lid in sorted(hooks.q.keys()):
            series.append((f"L{lid} Q", flatten_entries(hooks.q[lid])))
        for lid in sorted(hooks.k.keys()):
            series.append((f"L{lid} K", flatten_entries(hooks.k[lid])))

        hooks._remove()  # cleanup

        n_tokens = ids.shape[-1]
        plot_kdes(series, n_tokens, title_prefix="Empirical PDFs of Q/K (KDE) — Phi-2")

    except Exception as e:
        print(f"[Phi-2 not available or hook discovery failed ({e}). Switching to synthetic demo mode.]", file=sys.stderr)
        synthetic_demo(target_seq_len)

def synthetic_demo(n_tokens: int, d: int = 256, n_layers: int = 12):
    """
    Generates random Q/K with Gaussian entries (mean 0, variance 1/d) as a light-weight fallback.
    """
    rng = np.random.default_rng(0)
    series = []
    # All-layers pooled
    all_q = []
    all_k = []
    for lid in range(n_layers):
        Q = rng.normal(loc=0.0, scale=1.0/np.sqrt(d), size=(1, n_tokens, d))
        K = rng.normal(loc=0.0, scale=1.0/np.sqrt(d), size=(1, n_tokens, d))
        qv = Q.reshape(-1)
        kv = K.reshape(-1)
        series.append((f"L{lid} Q", qv))
        series.append((f"L{lid} K", kv))
        all_q.append(qv); all_k.append(kv)
    series.insert(0, ("K (all)", np.concatenate(all_k)))
    series.insert(0, ("Q (all)", np.concatenate(all_q)))

    plot_kdes(series, n_tokens, title_prefix="Empirical PDFs of Q/K (KDE) — Synthetic")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=2048, help="Number of tokens n.")
    parser.add_argument("--prompt", type=str, default="A" * 128, help="Seed prompt text.")
    parser.add_argument("--device", type=str, default="auto", help="'auto' or a CUDA device like 'cuda:0'.")
    args = parser.parse_args()
    run_with_phi2(args.prompt, args.seq_len, device=args.device)
