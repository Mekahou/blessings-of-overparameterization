# Packages
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
#-----------------------------------------------------------------------

# Importing functions from src/utils_rbc.py
from src.utils_rbc import (get_parameters,
                           steady_state,
                           state_space_bounds,
                           grid_kz,
                           gauss_hermite, NN_2D, count_parameters,
                           euler_residual, build_vfi_interpolator, simulate_path)
#-----------------------------------------------------------------------

# parameters & steady state values
params = get_parameters()
k_ss, c_ss = steady_state(params)
#------------------------------------------------------------------------

# Training data & data loader
num_k = 20
num_z = 20
kz_train, grid_k, grid_z = grid_kz(params, num_k=num_k, num_z=num_z)
data_loader = DataLoader(kz_train, batch_size=len(kz_train), shuffle=False)
#-----------------------------------------------------------------------

# Gauss-Hermite quadrature points and weights
zeta, w = gauss_hermite(M=15)
#-----------------------------------------------------------------------

# Value function approximator
policy_interp = build_vfi_interpolator(params)
#-----------------------------------------------------------------------

# Test data for evaluation 
dummy_nn = NN_2D() #dummy_nn is a placeholder — simulate_path requires an nn.Module argument, but we won't use it for evaluation
_, k_path_vfi_test, z_path_test = simulate_path(
    params, dummy_nn, policy_interp, k0=0.5*k_ss, T=29, mode='with_shocks', seed=1
)
kz_test = torch.tensor(
    np.column_stack([k_path_vfi_test, z_path_test]), dtype=torch.float32
) # Stack into (T, 2) tensor of (k_t, z_t) pairs for Euler residual evaluation
#-----------------------------------------------------------------------

# Training loop/function
def training_k(params, seed, dim_hidden, num_layers, hidden_activation,
               max_epochs=10001, tol=1e-8, lr=1e-3,
               step_size=100, gamma=0.99, penalty_weight=0.01):
    """
    Train a neural network policy on the Euler residual loss for the RBC model.

    Parameters
    ----------
    params           : dict from get_parameters()
    seed             : int  — controls NN initialization
    dim_hidden       : int  — number of hidden units per layer
    num_layers       : int  — number of hidden layers
    hidden_activation: nn.Module class (e.g. nn.ReLU, nn.Sigmoid)
    penalty_weight   : float — weight on steady-state penalty (k'(k_ss,0) - k_ss)^2

    Returns
    -------
    dict with keys:
        'n_params'    : total trainable parameters
        'train_loss'  : final MSE Euler residual on training grid
        'test_loss'   : MSE Euler residual on kz_test path
        'abs_rel_err' : array of absolute relative errors vs VFI along kz_test path
        'k_path_nn'   : capital path simulated under NN policy (T,)
        'k_path_vfi'  : capital path simulated under VFI policy (T,)
    """
    # ── Build network ──────────────────────────────────────────────────────
    policy_NN = NN_2D(dim_hidden=dim_hidden, layers=num_layers,
                      hidden_activation=hidden_activation, seed=seed)
    n_params = count_parameters(policy_NN)

    # ── Steady-state anchor ────────────────────────────────────────────────
    k_ss_val = steady_state(params)[0]
    ss_state = torch.tensor([[k_ss_val, 0.0]], dtype=torch.float32)
    k_ss_t   = torch.tensor(k_ss_val, dtype=torch.float32)

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(policy_NN.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ── Training loop ──────────────────────────────────────────────────────
    policy_NN.train()
    log_epochs = {0, 5000, max_epochs - 1}

    for epoch in range(max_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            residuals  = euler_residual(params, policy_NN, batch, zeta, w)
            euler_loss = (residuals ** 2).mean()
            ss_penalty = (policy_NN(ss_state) - k_ss_t) ** 2
            loss       = euler_loss + penalty_weight * ss_penalty
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch in log_epochs:
            with torch.no_grad():
                res_test_log  = euler_residual(params, policy_NN, kz_test, zeta, w)
                test_loss_log = (res_test_log ** 2).mean().item()
            print(f'epoch {epoch:>5d} | train loss: {euler_loss.item():.2e} | '
                  f'test loss: {test_loss_log:.2e} | ss_penalty: {ss_penalty.item():.2e}')

        # Early stopping on Euler loss only
        if euler_loss.item() < tol:
            if epoch not in log_epochs:
                with torch.no_grad():
                    res_test_log  = euler_residual(params, policy_NN, kz_test, zeta, w)
                    test_loss_log = (res_test_log ** 2).mean().item()
                print(f'epoch {epoch:>5d} | train loss: {euler_loss.item():.2e} | '
                      f'test loss: {test_loss_log:.2e} | ss_penalty: {ss_penalty.item():.2e}  [early stop]')
            break

    # ── Train loss (final pass over full training grid) ────────────────────
    policy_NN.eval()
    with torch.no_grad():
        res_train  = euler_residual(params, policy_NN, kz_train, zeta, w)
        train_loss = (res_train ** 2).mean().item()

    # ── Test loss (Euler residual on VFI-simulated path) ───────────────────
    with torch.no_grad():
        res_test  = euler_residual(params, policy_NN, kz_test, zeta, w)
        test_loss = (res_test ** 2).mean().item()

    # ── Relative error vs VFI along the same test path (seed=1) ───────────
    k_path_nn, k_path_vfi, _ = simulate_path(
        params, policy_NN, policy_interp,
        k0=0.5 * k_ss, T=29, mode='with_shocks', seed=1
    )
    abs_rel_err = np.abs(k_path_nn - k_path_vfi) / np.abs(k_path_vfi)

    return {
        'n_params'    : n_params,
        'train_loss'  : train_loss,
        'test_loss'   : test_loss,
        'abs_rel_err' : abs_rel_err,
        'k_path_nn'   : k_path_nn,
        'k_path_vfi'  : k_path_vfi,
    }

#----------------------------------------------------------------------------

if __name__ == "__main__":

    part_1 = np.arange(1, 100, 15)
    part_2 = np.arange(100, 400, 25)
    part_3 = np.arange(400, 550, 50)
    grid_dim_hidden = np.concatenate([part_1, part_2, part_3])

    bottom_quantile = 0.1
    top_quantile    = 0.9

    os.makedirs("results/rbc", exist_ok=True)
    # First Experiment: different seeds, one hidden layer, different widths, ReLU activation

    results_1_layer_relu = []

    for h in grid_dim_hidden:
        for seed in range(50):
            print(f"dim_hidden = {h}")
            print(f"seed = {seed}")
            out = training_k(
                params            = params,
                seed              = seed,
                dim_hidden        = h,
                num_layers        = 1,
                hidden_activation = nn.ReLU,
            )

            results_1_layer_relu.append({
                "dim_hidden"                : h,
                "seed"                      : seed,
                "n_params"                  : out["n_params"],
                "train_loss"                : out["train_loss"],
                "test_loss"                 : out["test_loss"],
                "abs_rel_err_median"        : np.median(out["abs_rel_err"]),
                "abs_rel_err_max"           : np.max(out["abs_rel_err"]),
                "abs_rel_err_10_percentile" : np.percentile(out["abs_rel_err"], bottom_quantile * 100),
                "abs_rel_err_90_percentile" : np.percentile(out["abs_rel_err"], top_quantile * 100),
            })

            # Save after every run — guards against kernel crash during long experiment
            pd.DataFrame(results_1_layer_relu).to_csv(
                "results/rbc/RBC_results_1_layer_relu.csv", index=False
            )
    # Second Experiment: fixed width, 512 , different seeds
    results_512 = []
    for seed in range(50):
        print(f"dim_hidden = 512 | seed = {seed}")
        out = training_k(
            params            = params,
            seed              = seed,
            dim_hidden        = 512,
            num_layers        = 1,
            hidden_activation = nn.ReLU,
        )

        results_512.append({
            "seed"       : seed,
            "n_params"   : out["n_params"],
            "train_loss" : out["train_loss"],
            "test_loss"  : out["test_loss"],
            "k_path_nn"  : out["k_path_nn"].tolist(),
            "k_path_vfi" : out["k_path_vfi"].tolist(),
        })

        pd.DataFrame(results_512).to_csv(
            "results/rbc/RBC_results_1_layer_relu_dim_hidden_512.csv", index=False
        )
    # third Experiment: fixed width, 32 , different seeds
    results_32 = []

    for seed in range(50):
        print(f"dim_hidden = 32 | seed = {seed}")
        out = training_k(
            params            = params,
            seed              = seed,
            dim_hidden        = 32,
            num_layers        = 1,
            hidden_activation = nn.ReLU,
        )

        results_32.append({
            "seed"       : seed,
            "n_params"   : out["n_params"],
            "train_loss" : out["train_loss"],
            "test_loss"  : out["test_loss"],
            "k_path_nn"  : out["k_path_nn"].tolist(),
            "k_path_vfi" : out["k_path_vfi"].tolist(),
        })

        pd.DataFrame(results_32).to_csv(
            "results/rbc/RBC_results_1_layer_relu_dim_hidden_32.csv", index=False
        )