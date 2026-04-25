https://github.com/user-attachments/assets/b589a5e9-cf63-4cb0-be96-02e480bbf38d

# The Blessings of Overparameterization: Applications in Solving Economic Models

This repository contains the code and results for the paper:

> **The Blessings of Overparameterization: Applications in Solving Economic Models**

We study how overparameterized neural networks behave when used to solve dynamic economic models via the Euler equation. Across three canonical models — Linear-Quadratic (LQ), McCall job search, and Real Business Cycle (RBC) — we document that larger networks systematically produce better approximations, consistent with the double descent phenomenon from the machine learning literature.

---

## Repository Structure

```
blessings/
├── src/
│   ├── utils_lq.py          # LQ model: Bellman, Euler, policy iteration
│   ├── utils_mccall.py      # McCall model: value function, reservation wage
│   └── utils_rbc.py         # RBC model: Euler residual, VFI, simulation
│
├── experiments/
│   ├── run_lq_policy.py              # LQ policy function experiment
│   ├── run_lq_policy_robustness.py   # LQ robustness: activations, architectures
│   ├── run_lq_value_function.py      # LQ value function experiment
│   ├── run_mccall.py                 # McCall main experiment
│   ├── run_mccall_robustness.py      # McCall robustness
│   ├── run_rbc.py                    # RBC main experiment
│   └── run_rbc_robustness.py         # RBC robustness
│
├── plots/
│   ├── plots_lq.py          # Generates all LQ figures
│   ├── plots_mccall.py      # Generates all McCall figures
│   └── plots_rbc.py         # Generates all RBC figures
│
├── results/
│   ├── lq/                  # CSV results for LQ experiments
│   ├── mccall/              # CSV results for McCall experiments
│   └── rbc/                 # CSV results for RBC experiments
│
├── figures/                 # Draft-ready PDF figures
├── generate_figures.py      # Runs all plot scripts in sequence
├── requirements.txt
└── README.md
```

---

## Models

### Linear-Quadratic (LQ)
A canonical optimal control problem with a known analytical solution. Used to validate the method and study the effect of network width on policy and value function approximation error.

### McCall Job Search
A discrete-time search model where an unemployed worker accepts or rejects wage offers. The value function has a kink at the reservation wage. We study how network width affects approximation of this non-smooth function.

### Real Business Cycle (RBC)
A standard stochastic growth model solved via the Euler equation. Networks are trained to satisfy Euler equation residuals on a grid over the state space $(k, z)$. We compare the learned capital path against a VFI benchmark across 50 random seeds and a wide range of network sizes.

---

## Installation

```bash
pip install -r requirements.txt
```

All experiments use plain Python scripts — no notebooks required for running experiments.

---

## Reproducing Results

### Step 1 — Run experiments

Each script in `experiments/` is self-contained and saves results as CSV files to `results/`.

> **Warning:** Running all experiments from scratch can take several days depending on your hardware. Results for all experiments are already committed to `results/` so figures can be reproduced without re-running.

```bash
python experiments/run_lq_policy.py
python experiments/run_lq_policy_robustness.py
python experiments/run_lq_value_function.py
python experiments/run_mccall.py
python experiments/run_mccall_robustness.py
python experiments/run_rbc.py
python experiments/run_rbc_robustness.py
```


### Step 2 — Generate figures

Run all plot scripts at once with:

```bash
python generate_figures.py
```

Or individually:

```bash
python plots/plots_lq.py
python plots/plots_mccall.py
python plots/plots_rbc.py
```

Figures are saved to `figures/`.

---

## Requirements

- Python 3.10+
- PyTorch 2.10
- NumPy, SciPy, pandas, matplotlib, quantecon

See `requirements.txt` for pinned versions.
