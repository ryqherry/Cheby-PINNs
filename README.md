# Chebyshev-Augmented One-Shot Transfer Learning for PINNs on Nonlinear Differential Equations

**Authors:** Yiqi Rao, Pavlos Protopapas  
**Contact:** herryrao@g.harvard.edu  
**Submission:** Under review at ICLR 2026 Workshop on AI & PDE

---

## Overview
Physics-Informed Neural Networks (PINNs) offer a flexible paradigm for solving differential equations by embedding governing laws into the training objective. A persistent limitation is instance specificity: standard PINNs typically require retraining for each new forcing term, boundary/initial condition, or parameter setting. One-shot transfer learning (OTL) addresses this bottleneck for linear operators by freezing a pretrained latent representation and computing optimal output weights in closed form, but for nonlinear problems closed-form adaptation is generally unavailable because the loss is nonconvex in the output layer.

To address such limitations, we extend one-shot transfer learning for PINNs to broader nonlinearities by approximating smooth nonlinear terms with truncated **Chebyshev polynomial surrogates**, then applying a **perturbative expansion** into a sequence of linear subproblems. A **multi-head PINN** is trained offline to learn a reusable latent space for a fixed dominant linear operator; online, new problem instances are solved via **closed-form output-layer solves** (no retraining of the network body).

This repository contains the implementation of the model and experiments developed in the paper.

---

## Repository Structure
```text
CHEBY-PINNS/
├─ chebypinns/
│  ├─ ODE/
│  │  ├─ experiments_ODE.ipynb             # experiments
│  │  ├─ helper.py                         # helper functions
│  │  ├─ model.py                          # model architecture
│  │  ├─ transfer.py                       # transfer learning module
│  │  ├─ model_ODE_cos.pickle
│  │  └─ model_ODE_inv_sq.pickle
│  └─ PDE/
│     ├─ reaction_diffusion_train.ipynb    # model training
│     ├─ reaction_diffusion_TL.ipynb       # experiments
│     ├─ reaction_diffusion_model.py       # model architecture
│     ├─ helper.py
│     ├─ transfer.py
│     ├─ Reaction_diffusion_16head_model_trig.pickle
│     ├─ Reaction_diffusion_16head_H_dict_trig.pickle
│     ├─ Reaction_diffusion_16head_log_trig.pickle
│     ├─ 4_heads_Reaction_Diffusion.png
│     └─ solutions_train_16heads.png
├─ pyproject.toml
└─ README.md
```

---

## Environment Setup

After cloning the repository, install the package in an virtual environment.

```bash
python -m venv <your_venv>
source <your_venv>/bin/activate
pip install -e .
```

---

## Acknowledgments
Part of the modules for model training and one-shot transfer learning was adapted from https://github.com/wanzhoulei/one_shot_pinn by Wanzhou Lei. We thank him for sharing the implementation.
