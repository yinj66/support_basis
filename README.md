# support_basis: Fast Attention Beyond Bounded Entries

This repository contains the reference implementation and experimental code accompanying the paper:

**Support Basis: Fast Attention Beyond Bounded Entries**
*Maryam Aliakbarpour, Vladimir Braverman, Junze Yin, Haochen Zhang*
(arXiv:2510.01643)
🎉 **Accepted as AISTATS 2026 Spotlight** 🎉

---

## Overview

> **AISTATS 2026 Spotlight** paper presenting the first provably fast attention approximation framework that goes beyond the bounded-entry assumption, while remaining compatible with modern Transformers.

Softmax attention is a core primitive in Transformer models, but its **quadratic time complexity** in the sequence length remains a major bottleneck for scaling large language models (LLMs). Recent theoretical work has shown that sub-quadratic attention is possible, but only under **strong bounded-entry assumptions** on the query (Q), key (K), and value (V) matrices—assumptions that are typically violated in real, trained models.

This paper introduces **support-basis decomposition**, a new framework that enables **provably fast attention approximation beyond bounded entries**, while remaining compatible with modern Transformer architectures.

This repository provides:

* Code for **computational complexity and runtime experiments** used in the paper.
* Scripts for **visualizing the empirical entry distributions of Q and K** across layers and models, validating the paper’s distributional assumptions.

---

## Background and Motivation

Prior work by Alman & Song (NeurIPS 2023) established a *time-optimal* sub-quadratic algorithm for approximating softmax attention. However, their result relies on a restrictive assumption:

> All entries of Q, K, and V must be bounded by ( o(\sqrt{\log n}) ).

Empirically, this assumption **does not hold** in modern Transformers. In practice, Q and K entries can be unbounded, and naive polynomial approximations either lose accuracy or become computationally infeasible.

This raises a fundamental question addressed by this work:

> *Can we approximate softmax attention efficiently and accurately under realistic, implementable assumptions—or even with no assumptions at all?*

---

## Key Ideas and Novelty

### 1. Support-Basis Decomposition

The central idea of the paper is to decompose the attention computation based on **entry magnitude**:

* **Large entries** (rare but influential) are handled *exactly*.
* **Small entries** (dense and well-behaved) are approximated using low-degree polynomial methods.

This decomposition yields a **support basis** that isolates the hard part of attention into a sparse structure, while keeping the remaining computation efficient.

### 2. Beyond the Bounded-Entry Assumption

* We empirically show that Q and K entries in real models (e.g., TinyLlama, OPT, Phi-2, LLaDA, ViT) closely follow **sub-Gaussian distributions**.
* Under this mild and realistic assumption, we prove a **sub-quadratic-time algorithm** with strong ( \ell_\infty ) error guarantees.

### 3. No-Assumption Guarantee via Multi-Threshold Bases

To go further, the paper introduces a **multi-threshold support basis** that removes *all* distributional assumptions:

* Q and K are partitioned into multiple magnitude buckets.
* Each bucket is approximated separately using polynomial attention.

This yields the first general sub-quadratic attention approximation algorithm that works for **arbitrary Q and K**, at the cost of a weaker (but unavoidable) accuracy dependence.

### 4. Theoretical Justification for Polynomial Attention

The framework provides a new theoretical explanation for why **polynomial attention** (e.g., replacing ( e^x ) with ( x^\beta )) works well in practice:

* Softmax attention can be decomposed into a *sum of polynomial attentions*.
* Sketching techniques further reduce the runtime while preserving accuracy.

This offers the first principled justification for the empirical success of recent polynomial-attention methods.

---

## Repository Structure

* `supp_basis_optimization.py`
  Implements the core runtime and complexity experiments for support-basis–based attention approximation, including polynomial feature construction and timing benchmarks.

* `visualize_llada.py`
  Visualizes empirical KDEs of Q and K entries across layers for **LLaDA-8B**, highlighting concentration and tail behavior.

* `visualize_phi_2.py`
  Extracts and plots Q/K entry distributions for **Phi-2**, handling both fused and unfused QKV projections.

* `visualize_tiny_llama.py`
  Produces per-layer visualizations of Q and K distributions for **TinyLlama**, with reference thresholds at ( \pm \sqrt{\log n} ).

* `visualize_ViT.py`
  Extends the analysis beyond language models, visualizing Q/K/V entry distributions in **Vision Transformers (ViT)**.

---

## Reproducibility Notes

* All visualization scripts rely on forward hooks to capture *pre-RoPE* Q and K entries.
* KDE plots include reference thresholds used in the theoretical analysis.
* Experiments are designed to be lightweight and illustrative, as the paper’s primary contribution is theoretical.

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{aliakbarpour2025support,
  title={Support Basis: Fast Attention Beyond Bounded Entries},
  author={Aliakbarpour, Maryam and Braverman, Vladimir and Yin, Junze and Zhang, Haochen},
  journal={arXiv preprint arXiv:2510.01643},
  year={2025}
}
```

---

## Contact

For questions or clarifications, feel free to reach out to the authors or open an issue.
