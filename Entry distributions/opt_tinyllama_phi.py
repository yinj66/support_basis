
import os
import re
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# =================
# Config
# =================
MODEL_IDS = [
    'facebook/opt-1.3b',
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
]

DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT = (
    "In the fast-paced world of modern research and technology, the ability to adapt quickly to new knowledge, integrate interdisciplinary ideas, and communicate them clearly has become a defining factor for success, not only for academic researchers but also for professionals working in industry, policy, or creative sectors. The challenge is no longer simply about having access to information—since the digital age has democratized knowledge to an unprecedented degree—but rather about cultivating the skills necessary to filter, evaluate, and synthesize the vast amount of data that is constantly flowing around us. A researcher today may begin the morning reading about advances in large language models, spend the afternoon designing experiments to validate theoretical insights, and finish the day by considering applications in medicine, finance, or education. This fluid movement between levels of abstraction demands both intellectual flexibility and a strong methodological foundation. At the same time, collaboration has emerged as a core feature of progress: breakthroughs increasingly arise not from the lone genius archetype, but from teams that combine different strengths, whether it is the theoretical rigor of mathematicians, the practical engineering sense of computer scientists, the domain knowledge of biologists, or the design intuition of human-computer interaction experts. Alongside collaboration, communication is equally essential. A brilliant idea poorly explained is often a wasted opportunity, while even a moderately novel insight presented with clarity and precision can influence the trajectory of a field. In this context, the role of writing, speaking, and visualizing cannot be underestimated. Writing a paper or report is not just a matter of documenting results but of shaping the interpretation and framing of those results, guiding how others will build upon them. Similarly, presenting at conferences, creating effective figures, and explaining technical ideas to broader audiences are all skills that expand the impact of one’s work beyond immediate circles. Another layer to this landscape is the increasing pressure to balance depth with breadth. Specialization is necessary to push the frontier of a subfield, yet the most impactful research often arises from unexpected connections, such as applying methods from signal processing to neuroscience or adapting optimization techniques from physics to machine learning. To navigate this dual demand, researchers must cultivate a meta-skill: the ability to learn how to learn efficiently, to enter new fields without being overwhelmed, and to identify the key assumptions, tools, and open questions that define them. Underlying all of this is resilience and persistence, since research inevitably involves setbacks, failed experiments, and long periods of uncertainty. The process is rarely linear; rather, it is iterative and recursive, resembling more a spiral of refinement than a straight path toward discovery. In the end, success in modern research and professional life lies in the interplay of curiosity, rigor, creativity, and communication—qualities that allow individuals and teams not only to generate knowledge but to ensure that this knowledge becomes meaningful, usable, and transformative in the broader world."
)

OUTDIR = "entry_distribution_plots"
os.makedirs(OUTDIR, exist_ok=True)

# =================
# Helpers
# =================
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def register_hooks(model):
    """
    Store per-layer Q/K activations.
    Each entry in acts['q'] or acts['k'] is expected to be one layer's output.
    """
    acts = {"q": [], "k": []}
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)

    def save(name):
        def fn(module, _, out):
            if isinstance(out, tuple):
                out = out[0]
            acts[name].append(out.detach().cpu().reshape(-1))
        return fn

    def save_fused():
        def fn(module, _, out):
            if isinstance(out, tuple):
                out = out[0]
            with torch.no_grad():
                y = out.detach()
                last = y.shape[-1]

                q_dim = d_model if d_model is not None else last // 3
                kv_dim = (last - q_dim) // 2

                if q_dim <= 0 or kv_dim <= 0 or q_dim + 2 * kv_dim != last:
                    return

                q, k, _ = torch.split(y, [q_dim, kv_dim, kv_dim], dim=-1)
                acts["q"].append(q.cpu().reshape(-1))
                acts["k"].append(k.cpu().reshape(-1))
        return fn

    for name, module in model.named_modules():
        low = name.lower()

        # Separate Q/K projections
        if "q_proj" in low:
            module.register_forward_hook(save("q"))
        elif "k_proj" in low:
            module.register_forward_hook(save("k"))

        # GPT-style fused QKV
        elif "c_attn" in low:
            module.register_forward_hook(save_fused())

        # Your custom fused projection name
        elif "att_proj" in low:
            module.register_forward_hook(save_fused())

    return acts

def flatten_per_layer(tensor_list):
    return [x.reshape(-1).numpy() for x in tensor_list]

def get_layerwise_colors(n_layers):
    # enough distinct colors for many layers
    cmap1 = plt.get_cmap("tab20")
    cmap2 = plt.get_cmap("tab20b")
    cmap3 = plt.get_cmap("tab20c")
    colors = []
    for cmap in [cmap1, cmap2, cmap3]:
        for i in range(cmap.N):
            colors.append(cmap(i))
    return colors[:n_layers]

def plot_layerwise_qk_distribution(
    q_layers,
    k_layers,
    model_name,
    seq_len,
    save_path,
    max_points_per_layer=80000,
    num_grid=1200,
    x_percentile_low=0.2,
    x_percentile_high=99.8,
):
    """
    Plot per-layer KDE curves:
      - same color for same layer
      - solid for Query
      - dashed for Key
    """
    n_layers = min(len(q_layers), len(k_layers))
    if n_layers == 0:
        print(f"No layerwise Q/K found for {model_name}")
        return

    q_layers = q_layers[:n_layers]
    k_layers = k_layers[:n_layers]

    # determine plotting range robustly
    all_vals = np.concatenate(q_layers + k_layers)
    x_low = np.percentile(all_vals, x_percentile_low)
    x_high = np.percentile(all_vals, x_percentile_high)
    xs = np.linspace(x_low, x_high, num_grid)

    colors = get_layerwise_colors(n_layers)

    fig, ax = plt.subplots(figsize=(11, 7))

    legend_handles = []
    legend_labels = []

    for i, (q, k) in enumerate(zip(q_layers, k_layers)):
        color = colors[i]

        # subsample if too many points
        q_sample = q if len(q) <= max_points_per_layer else np.random.choice(q, max_points_per_layer, replace=False)
        k_sample = k if len(k) <= max_points_per_layer else np.random.choice(k, max_points_per_layer, replace=False)

        # KDE
        try:
            q_kde = gaussian_kde(q_sample)
            k_kde = gaussian_kde(k_sample)

            line_q, = ax.plot(xs, q_kde(xs), color=color, linewidth=1.8, linestyle="-")
            line_k, = ax.plot(xs, k_kde(xs), color=color, linewidth=1.8, linestyle="--")

            legend_handles.extend([line_q, line_k])
            legend_labels.extend([f"Layer {i}: Query", f"Layer {i}: Key"])
        except Exception as e:
            print(f"KDE failed at layer {i} for {model_name}: {e}")

    # threshold lines ±sqrt(log n)
    thr = math.sqrt(math.log(seq_len)) if seq_len > 1 else 0.0
    ax.axvline(-thr, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(thr, color="red", linestyle="--", linewidth=1.2)

    ymax = ax.get_ylim()[1]
    ax.text(-thr, ymax * 0.92, r"$-\sqrt{\log(n)}$", color="red", ha="left", va="top", fontsize=9)
    ax.text(thr, ymax * 0.92, r"$\sqrt{\log(n)}$", color="red", ha="left", va="top", fontsize=9)

    ax.set_title(model_name, fontsize=16)
    ax.set_xlabel("Value", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)

    ax.set_xlim(-30, 30)

    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.25)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=2,
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        handlelength=2.5,
        columnspacing=1.0,   # spacing between columns
        labelspacing=0.3,    # spacing between rows
    )

    plt.tight_layout(rect=[0, 0, 0.77, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# =================
# Main
# =================
for MODEL_ID in MODEL_IDS:
    print(f"\n===== Running {MODEL_ID} =====")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    activations = register_hooks(model)

    inputs = tok(TEXT, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        _ = model(**inputs)

    q_layers = flatten_per_layer(activations["q"])
    k_layers = flatten_per_layer(activations["k"])

    print(f"Captured {len(q_layers)} Q layers and {len(k_layers)} K layers.")

    save_path = os.path.join(
        OUTDIR,
        f"{safe_name(MODEL_ID)}_layerwise_qk_distribution.pdf"
    )

    plot_layerwise_qk_distribution(
        q_layers=q_layers,
        k_layers=k_layers,
        model_name=MODEL_ID.split("/")[-1],
        seq_len=seq_len,
        save_path=save_path,
    )

    print(f"Saved to: {save_path}")