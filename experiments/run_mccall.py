# Packages
import sys, os      
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
                                                                                                   
#--------------------------------------------------------------------

# Importing functions from src/utils_mccall.py 
from src.utils_mccall import (
    get_parameters, 
    v_theory, 
    grid_w, 
    NN, 
    count_parameters,
    bellman_residual
) 
#--------------------------------------------------------------------

# setting the parameters

param = get_parameters()
#--------------------------------------------------------------------

# Training data & test data & dataloader & closed form solution on test data
grid_num_train = 11
w_train = grid_w(params = param, grid_num = grid_num_train)
data_loader = DataLoader(w_train, batch_size=len(w_train), shuffle=False) #full batch


grid_num_test = 100
w_test = grid_w(params = param, grid_num = grid_num_test)

v_theory_test  = v_theory(w=w_test,  params=param)
#--------------------------------------------------------------------

# quantiles 
bottom_quantile = 0.1
top_quantile = 0.9

#--------------------------------------------------------------------

# Training the neural network
def training_v(params, seed, dim_hidden, num_layers, 
               hidden_activation,max_epochs=10001, 
               tol=1e-8, lr=1e-2, step_size=100, gamma=0.99):
    """
    Train a neural network value function on the Bellman residual loss
    for the McCall job search model.

    Parameters
    ----------
    params            : dict from get_parameters()
    seed              : int   — controls NN initialization
    dim_hidden        : int   — number of hidden units
    num_layers        : int   — number of hidden layers
    hidden_activation : nn.Module class (e.g. nn.ReLU, nn.LeakyReLU)
    max_epochs        : int   — maximum number of training epochs
    tol               : float — early stopping threshold on training loss
    lr                : float — initial Adam learning rate
    step_size         : int   — StepLR step size (epochs per decay)
    gamma             : float — StepLR decay factor

    Returns
    -------
    dict with keys:
        "param_num"        : total trainable parameters
        "abs_rel_err_test" : absolute relative errors vs closed-form on test grid
        "v_nn"             : NN value function predictions on test grid (T,)
        "v_theory"         : closed-form value function on test grid (T,)
        "w_test"           : test grid wage values (T,)
        "train_loss"       : final Bellman MSE on training grid
        "test_loss"        : Bellman MSE on test grid
    """
    v_theta   = NN(layers=num_layers, dim_hidden=dim_hidden,
                   hidden_activation=hidden_activation, seed=seed)
    optimizer = torch.optim.Adam(v_theta.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(max_epochs):
        for w in data_loader:
            optimizer.zero_grad()
            residual = bellman_residual(params=params, v_theta=v_theta, w=w)
            loss     = (residual ** 2).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10000 == 0:
            print(f"epoch = {epoch}, loss = {loss.detach().numpy():.2e}")

        if loss.item() < tol:
            print(f"Early stopping at epoch {epoch}, loss = {loss.detach().numpy():.2e}")
            break

    # ── Evaluation ───────────────────────────────────────────────────────────
    with torch.no_grad():
        v_theta_test     = v_theta(w_test)
        abs_rel_err_test = torch.abs((v_theory_test - v_theta_test) / v_theory_test)

        train_loss    = loss.item()
        residual_test = bellman_residual(params=params, v_theta=v_theta, w=w_test)
        test_loss     = (residual_test ** 2).mean().item()

    return {
        "param_num"       : count_parameters(v_theta),
        "abs_rel_err_test": abs_rel_err_test,
        "v_nn"            : v_theta_test.squeeze().numpy(),
        "v_theory"        : v_theory_test.squeeze().numpy(),
        "w_test"          : w_test.squeeze().numpy(),
        "train_loss"      : train_loss,
        "test_loss"       : test_loss,
    }

#----------------------------------------------------------------------------

if __name__ == "__main__":

    # Running the experiments
    part_1 = np.arange(1, 20)
    part_2 = np.arange(20, 50, 5)
    part_3 = np.arange(50, 400, 20)
    grid_dim_hidden = np.concatenate([part_1, part_2, part_3])

    os.makedirs("results/mccall", exist_ok=True)

    # First Experiment: different seeds, one hidden layer, different widths, ReLU activation
    results_value_function_1_layer_relu = []

    for h in grid_dim_hidden:
        for seed in range(50):
            print(f"dim_hidden = {h}")
            print(f"seed = {seed}")
            out = training_v(
                params            = param,
                seed              = seed,
                dim_hidden        = h,
                num_layers        = 1,
                hidden_activation = nn.ReLU
            )

            results_value_function_1_layer_relu.append({
                "dim_hidden"                                   : h,
                "seed"                                         : seed,
                "param_num_v"                                  : out["param_num"],
                "train_loss_bellman"                           : out["train_loss"],
                "test_loss_bellman"                            : out["test_loss"],
                "abs_rel_err_test_value_function_median"       : out["abs_rel_err_test"].median().item(),
                "abs_rel_err_test_value_function_max"          : out["abs_rel_err_test"].max().item(),
                "abs_rel_err_test_value_function_10_percentile": out["abs_rel_err_test"].quantile(bottom_quantile).item(),
                "abs_rel_err_test_value_function_90_percentile": out["abs_rel_err_test"].quantile(top_quantile).item(),
            })

            # Save after every run — guards against kernel crash during long experiment
            pd.DataFrame(results_value_function_1_layer_relu).to_csv(
                "results/mccall/McCall_results_1_layer_relu.csv", index=False
            )
    # Second Experiment: fixed width, 512 , different seeds
    results_512 = []

    for seed in range(50):
        print(f"dim_hidden = 512 | seed = {seed}")
        out = training_v(
            params            = param,
            seed              = seed,
            dim_hidden        = 512,
            num_layers        = 1,
            hidden_activation = nn.ReLU,
            )

        results_512.append({
            "seed"       : seed,
            "param_num"  : out["param_num"],
            "train_loss" : out["train_loss"],
            "test_loss"  : out["test_loss"],
            "v_nn"       : out["v_nn"].tolist(),
            "v_theory"   : out["v_theory"].tolist(),
            "w_test"     : out["w_test"].tolist(),
        })

        pd.DataFrame(results_512).to_csv(
            "results/mccall/McCall_results_1_layer_relu_dim_hidden_512.csv", index=False
            )

    # Third Experiment: fixed width, 24 , different seeds
    results_24 = []

    for seed in range(50):
        print(f"dim_hidden = 24 | seed = {seed}")
        out = training_v(
            params            = param,
            seed              = seed,
            dim_hidden        = 24,
            num_layers        = 1,
            hidden_activation = nn.ReLU,
        )

        results_24.append({
            "seed"       : seed,
            "param_num"  : out["param_num"],
            "train_loss" : out["train_loss"],
            "test_loss"  : out["test_loss"],
            "v_nn"       : out["v_nn"].tolist(),
            "v_theory"   : out["v_theory"].tolist(),
            "w_test"     : out["w_test"].tolist(),
        })

        pd.DataFrame(results_24).to_csv(
            "results/mccall/McCall_results_1_layer_relu_dim_hidden_24.csv", index=False
        )