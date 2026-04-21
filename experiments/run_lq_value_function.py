# Packages
import sys, os      
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
#--------------------------------------------------------------------

# Importing functions from src/utils_lq.py 
from src.utils_lq import (
    get_parameters, LQ_theory, grid_x, test_data_v,
    bellman_residual, NN_2D, count_parameters
)
# Importing training_u from run_lq_policy.py
from experiments.run_lq_policy import training_u
#--------------------------------------------------------------------

# setting the parameters

param = get_parameters()
#--------------------------------------------------------------------

# Obtaining the approximate policy
policy_results = training_u(params = param, seed = 123, dim_hidden = 8 , num_layers = 1, hidden_activation = nn.ReLU)
u_theta_star_nn = policy_results["optimal_policy"]
#--------------------------------------------------------------------

# Setting the data: test and train 
grid_num_y    = 12
grid_num_Y    = 12
y_upper_bound = 1.2
Y_upper_bound = 0.5
time_period   = 20

## Training data & data loader (built once, reused across all runs)
yY_train, _, _ = grid_x(params=param, grid_num_y=grid_num_y, grid_num_Y=grid_num_Y,
                         y_upper_bound=y_upper_bound, Y_upper_bound=Y_upper_bound)
data_loader_v = DataLoader(yY_train, batch_size=len(yY_train), shuffle=False)

## Test data
yY_test, y_hat_test, Y_test = test_data_v(params=param, time_period=time_period)         # Test path: simulate (y_hat, Y) using the true policy

#--------------------------------------------------------------------

# Training the value function
def training_v(params, u_theta, seed, dim_hidden, num_layers, hidden_activation,
               max_epochs=10001, tol=1e-8, lr=1e-3,
               step_size=100, gamma=0.99):
    """
    Train a neural network value function on the Bellman residual loss
    for the LQ model, given a fixed (pre-trained) policy u_theta.

    Parameters
    ----------
    params            : dict from get_parameters()
    u_theta           : trained policy nn.Module (in eval mode)
    seed              : int   — controls NN initialization
    dim_hidden        : int   — number of hidden units
    num_layers        : int   — number of hidden layers
    hidden_activation : nn.Module class (e.g. nn.ReLU, nn.Sigmoid)
    max_epochs        : int   — maximum number of training epochs
    tol               : float — early stopping threshold on training loss
    lr                : float — initial Adam learning rate
    step_size         : int   — StepLR step size (epochs per decay)
    gamma             : float — StepLR decay factor

    Returns
    -------
    dict with keys:
        'param_num'                                : total trainable parameters
        'abs_rel_err_test'                         : per-point absolute relative errors
        'v_nn'                                     : NN value predictions on test path (T,)
        'v_theory'                                 : closed-form value on test path (T,)
        'yY_test'                                  : test path (y_hat, Y) array (T, 2)
        'train_loss_bellman'                       : final Bellman MSE on training grid
        'test_loss_bellman'                        : Bellman MSE on test path
        'abs_rel_err_test_median'                  : median absolute relative error
        'abs_rel_err_test_max'                     : max absolute relative error
    """
    # ── Network and optimizer ────────────────────────────────────────────────
    v_theta   = NN_2D(layers=num_layers, dim_hidden=dim_hidden,
                      hidden_activation=hidden_activation, seed=seed)
    optimizer = torch.optim.Adam(v_theta.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(max_epochs):
        for yY_batch in data_loader_v:
            optimizer.zero_grad()
            residual = bellman_residual(params=params, u_theta=u_theta,
                                        v_theta=v_theta, yY=yY_batch)
            loss     = (residual ** 2).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10000 == 0:
            print(f"epoch = {epoch}, loss_Bellman = {loss.item():.2e}")

        if loss.item() < tol:
            print(f"Early stopping at epoch {epoch}, loss_Bellman = {loss.item():.2e}")
            break

    # ── Evaluation ───────────────────────────────────────────────────────────
    v_theta.eval()
    with torch.no_grad():
        train_loss = max(loss.item(), tol)

        v_theory_test, _ = LQ_theory(y=y_hat_test, Y=Y_test, params=params)

        res_test  = bellman_residual(params=params, u_theta=u_theta,
                                     v_theta=v_theta, yY=yY_test)
        test_loss = max((res_test ** 2).mean().item(), tol)

        v_hat       = v_theta(yY_test)
        abs_rel_err = torch.abs((v_hat - v_theory_test) / v_theory_test)

    return {
        'param_num'              : count_parameters(v_theta),
        'abs_rel_err_test'       : abs_rel_err,
        'v_nn'                   : v_hat.squeeze().detach().numpy(),
        'v_theory'               : v_theory_test.squeeze().detach().numpy(),
        'yY_test'                : yY_test.detach().numpy(),
        'train_loss_bellman'     : train_loss,
        'test_loss_bellman'      : test_loss,
        'abs_rel_err_test_median': abs_rel_err.median().item(),
        'abs_rel_err_test_max'   : abs_rel_err.max().item(),
    }

# quantiles 
bottom_quantile = 0.1
top_quantile = 0.9
#--------------------------------------------------------------------

if __name__ == "__main__":
    # Running the experiments: across different nodes and different seeds
    os.makedirs("results/lq", exist_ok=True)
    part_1 = np.arange(1, 100, 20)
    part_2 = np.arange(100, 500, 50)
    part_3 = np.arange(500, 1500, 100)
    grid_dim_hidden_v = np.concatenate([part_1, part_2, part_3])

    results_value_function_1_layer_relu = []

    for h in grid_dim_hidden_v:
        for seed in range(50):
            print(f"dim_hidden = {h} | seed = {seed}")
            out = training_v(
                params            = param,
                u_theta           = u_theta_star_nn,
                seed              = seed,
                dim_hidden        = h,
                num_layers        = 1,
                hidden_activation = nn.ReLU
            )

            results_value_function_1_layer_relu.append({
                'dim_hidden'                                    : h,
                'seed'                                          : seed,
                'param_num'                                     : out['param_num'],
                'train_loss_bellman'                            : out['train_loss_bellman'],
                'test_loss_bellman'                             : out['test_loss_bellman'],
                'abs_rel_err_test_median'                       : out['abs_rel_err_test_median'],
                'abs_rel_err_test_max'                          : out['abs_rel_err_test_max'],
                'abs_rel_err_test_10_percentile'                : out['abs_rel_err_test'].quantile(bottom_quantile).item(),
                'abs_rel_err_test_90_percentile'                : out['abs_rel_err_test'].quantile(top_quantile).item(),
            })

            # Save after every run — guards against kernel crash during long experiment
            pd.DataFrame(results_value_function_1_layer_relu).to_csv(
                'results/lq/LQ_results_value_1_layer_relu.csv', index=False
            )


