# Support Basis: Fast Attention Beyond Bounded Entries

**Maryam Aliakbarpour · Vladimir Braverman · Junze Yin · Haochen Zhang**  
*AISTATS 2026 Spotlight*

📄 Paper: *Support Basis: Fast Attention Beyond Bounded Entries*  
🔗 Code: https://github.com/yinj66/support_basis  

---

## Overview

Softmax attention has quadratic complexity in sequence length \( n \), which limits the scalability of large language models (LLMs). Prior sub-quadratic algorithms rely on the **bounded-entry assumption**, which rarely holds in practice.

This repository implements **Support-Basis Decomposition**, a new framework that:

- Removes the bounded-entry assumption  
- Achieves sub-quadratic runtime  
- Matches the error guarantees of prior work  


The repository includes:

- 🔬 Computational efficiency experiments  
- 📊 Entry distribution visualizations  
- 🧠 Downstream benchmark evaluations  
- 🤖 Modified LLaDA model implementation  

---

# Repository Structure

```
├── Computational efficiency experiment
│   └── supp_basis_optimization.py
├── Downstream task experiment
│   ├── generate.py
│   ├── run_all.sh
│   ├── run_chebyshev_ARC_challenge.py
│   ├── run_chebyshev_ARC_easy.py
│   ├── run_chebyshev_GPQA.py
│   ├── run_chebyshev_hellaswag.py
│   └── run_chebyshev_llmu.py
├── Entry distributions
│   ├── visualize_ViT.py
│   ├── visualize_llada.py
│   ├── visualize_phi_2.py
│   └── visualize_tiny_llama.py
├── LLaDA-8B-Instruct-B&S-light
│   ├── modeling_llada.py
│   ├── configuration_llada.py
│   ├── config.json
│   └── ...
└── README.md
```

---

# 1️⃣ Computational Efficiency Experiment

File:
```
Computational efficiency experiment/supp_basis_optimization.py
```

This script implements:

- Exact attention  
- Chebyshev polynomial approximation  
- Hybrid support-basis attention  
- Runtime comparison  

To run:

```bash
cd "Computational efficiency experiment"
python supp_basis_optimization.py
```

You may modify:

- `N` (sequence length)  
- `d` (hidden dimension)  
- `DEGREE` (Chebyshev polynomial degree)  

---

# 2️⃣ Entry Distribution Visualization

We empirically validate that Q and K entries behave sub-Gaussian.

Supported models:

- TinyLlama  
- Phi-2  
- LLaDA-8B  
- ViT  

Scripts:

```
Entry distributions/
├── visualize_tiny_llama.py
├── visualize_phi_2.py
├── visualize_llada.py
└── visualize_ViT.py
```

Each script:

- Extracts Q/K entries  
- Plots entry distributions  
- Draws ±√(log n) thresholds  

Example:

```bash
cd "Entry distributions"
python visualize_tiny_llama.py
```

These experiments support the theoretical assumption that large entries are rare.

---

# 3️⃣ Downstream Task Experiments

Directory:

```
Downstream task experiment/
```

We evaluate hybrid Chebyshev attention on multiple benchmarks.

## Datasets

- ARC-Challenge  
- ARC-Easy  
- GPQA  
- HellaSwag  
- MMLU  

Each script modifies the LLaDA model configuration:

```python
model.config.hybrid_exact_ratio = args.exact_ratio
model.config.hybrid_chebyshev_degree = args.chebyshev_degree
```

---

## Example: Run ARC Challenge

```bash
python run_chebyshev_ARC_challenge.py \
  --model_name "path/to/LLaDA-8B-Instruct-B&S" \
  --exact_ratio 0.2 \
  --chebyshev_degree 4
```

---

## Run All Benchmarks

```bash
bash run_all.sh
```

This runs:

- Degree 4 and 6  
- Exact ratio = 0.2  
- All downstream tasks  

---

## Parameters

| Parameter | Meaning |
|-----------|----------|
| `exact_ratio` | Fraction of entries computed exactly |
| `chebyshev_degree` | Polynomial degree for dense part |
| `steps` | Reverse diffusion steps |
| `block_length` | Semi-autoregressive block size |

---

# 4️⃣ LLaDA-8B Hybrid Model

Directory:

```
LLaDA-8B-Instruct-B&S-light/
```

Contains:

- Modified attention implementation  
- Hybrid support-basis logic  
- Config hooks  

Key parameters:

```python
model.config.hybrid_exact_ratio
model.config.hybrid_chebyshev_degree
```

These control:

- Single-threshold support basis  
- Polynomial degree  
- Exact computation sparsity  

---

# Algorithm Summary

## Single-Threshold Support Basis

1. Split Q, K into:
   - Small entries (dense)  
   - Large entries (sparse)  

2. Compute:
   - Exact attention on sparse support  
   - Polynomial approximation on dense support  

3. Combine via:
\[
\exp(A^{(s)}) + \exp(A^{(L)}) - \textbf{1}_{n \times n}
\]

---


# Dependencies

Install:

```bash
pip install torch transformers datasets scipy matplotlib tqdm
```

For GPU experiments:

- CUDA 11+  
- PyTorch ≥ 2.0 recommended  

---

# Reproducing Paper Results

### Runtime Experiments
```
Computational efficiency experiment/
```

### Entry Distributions
```
Entry distributions/
```

### Downstream Accuracy
```
Downstream task experiment/
```

---

# Citation

If you use this work, please cite:

```bibtex
@inproceedings{aliakbarpour2026support,
title={Support Basis: Fast Attention Beyond Bounded Entries},
author={Maryam Aliakbarpour and Vladimir Braverman and Junze Yin and Haochen Zhang},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=IgpnZIGFsD}
}
```


