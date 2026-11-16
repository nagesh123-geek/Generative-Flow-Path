# Generative Flow Models: Four Interpolation Probability Paths

This repository contains the implementation of four interpolation-based probability paths for Generative Flow Models. It includes training and inference scripts, trajectory analysis, velocity-score computation, and statistical analysis of generated distributions.

The framework explores four probability interpolation paths:

1. **Linear**
2. **Operator**
3. **Trigonometric**
4. **Spherical**

The implementations are evaluated using two datasets: a three-well potential system and molecular dynamics data from Trp-cage.

---

## Repository Structure

```text
Generative-Flow-Path/
│
├── data/
│   ├── threewell_data_downsampled.npy
│   └── 10_DATA_TRPCAGE.npy
│
├── source/
│   ├── flow_common.py
│   │
│   ├── train_linear.py
│   ├── train_operator.py
│   ├── train_trigonometric.py
│   ├── train_spherical.py
│   │
│   ├── infer_linear.py
│   ├── infer_operator.py
│   ├── infer_trigonometric.py
│   └── infer_spherical.py
│
├── velocity_score/
│   ├── velocity_score_computation.py
│   └── velocity_score_analysis.ipynb
│
└── mean_var/
    ├── flow_common.py
    ├── mean_var_trpcage.py
    └── run_all.sh
```

---

## 1. Dataset

The `data/` directory contains two datasets used for training and evaluation.

| Dataset | Description |
|---|---|
| `threewell_data_downsampled.npy` | Downsampled training data from a three-well potential system. |
| `10_DATA_TRPCAGE.npy` | Trp-cage molecular dynamics dataset containing 50,000 samples and 38 backbone dihedral features (10% of the available data). |

**Trp-cage data shape:** `(50000, 38)`

---

## 2. Source Code

The `source/` directory contains the common model backbone and the training and inference implementations for all four probability paths.

### Backbone

**`flow_common.py`**

Contains the common model architecture and probability path implementations shared across the different interpolation methods.

### Training

| Script | Description |
|---|---|
| `train_linear.py` | Training using the linear interpolation path. |
| `train_operator.py` | Training using the operator interpolation path. |
| `train_trigonometric.py` | Training using the trigonometric interpolation path. |
| `train_spherical.py` | Training using the spherical interpolation path. |

### Inference

| Script | Description |
|---|---|
| `infer_linear.py` | Sampling using the trained linear flow model. |
| `infer_operator.py` | Sampling using the trained operator flow model. |
| `infer_trigonometric.py` | Sampling using the trained trigonometric flow model. |
| `infer_spherical.py` | Sampling using the trained spherical flow model. |

---

## 3. Velocity Score Analysis

The `velocity_score/` directory contains scripts for analyzing the generated trajectories and evaluating their dynamical properties.

- **`velocity_score_computation.py`**: Computes and saves relevant statistical quantities, including KL divergence and the expectation of the velocity-score product.
- **`velocity_score_analysis.ipynb`**: Notebook for analyzing and visualizing the computed quantities.

---

## 4. Mean and Variance Analysis

The `mean_var/` directory contains scripts for computing the temporal evolution of statistical properties along the generated trajectories.

- **`mean_var_trpcage.py`**: Computes the mean and variance over time for a specified interpolation path.
- **`run_all.sh`**: Executes the mean and variance computation for all four interpolation paths.

To execute the analysis:

```bash
cd mean_var
bash run_all.sh
```

---

## 5. Workflow

The general workflow is:

1. Load the training dataset.
2. Select one of the four interpolation probability paths.
3. Train the corresponding generative flow model.
4. Generate samples and intermediate trajectories using the trained model.
5. Compute velocity-score-related statistical quantities.
6. Analyze the temporal evolution of mean and variance.
7. Compare the behavior of the four interpolation paths.

---

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/nagesh123-geek/Generative-Flow-Path.git
cd Generative-Flow-Path
```

Install the required Python dependencies:

```bash
pip install numpy torch matplotlib scipy jupyter
```

Additional dependencies may be required depending on the analysis scripts.

---

## 7. Research Scope

This repository provides a comparative implementation of four interpolation probability paths for generative flow models, with applications to:

- Multimodal probability distributions.
- Molecular dynamics data generation.
- Probability distribution evolution.
- Trajectory-level statistical analysis.
- KL divergence and velocity-score analysis.
- Temporal evolution of statistical moments.

The primary objective is to investigate how different interpolation paths influence the generative dynamics and the evolution of probability distributions.
