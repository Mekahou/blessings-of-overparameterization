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
    get_parameters, LQ_theory, grid_x, test_data_u,
    euler_residual, NN_1D, count_parameters
)
#--------------------------------------------------------------------

# setting the parameters

param = get_parameters()
#--------------------------------------------------------------------

# Training the neural network

def training_u(params, seed, dim_hidden, num_layers, hidden_activation,
               max_epochs=10001, tol=1e-8, lr=1e-3,
               step_size=100, gamma=0.99):
    """
    Train a neural network policy on the Euler residual loss
    for the LQ model.

    Parameters
    ----------
    params            : dict from get_parameters()
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
        'optimal_policy'   : trained NN model (nn.Module, in eval mode)
        'param_num'        : total trainable parameters
        'abs_rel_err_test' : absolute relative errors vs closed-form on test path
        'u_nn'             : NN policy predictions on test path (T,)
        'u_theory'         : closed-form policy on test path (T,)
        'Y_test'           : test path Y values (T,)
        'train_loss_euler' : final Euler MSE on training grid
        'test_loss_euler'  : Euler MSE on test path
    """
    # ── Training grid parameters ─────────────────────────────────────────────
    grid_num_y    = 12
    grid_num_Y    = 6
    y_upper_bound = 1.2
    Y_upper_bound = 0.5
    time_period   = 29

    # ── Build training data ──────────────────────────────────────────────────
    _, _, Y_train = grid_x(params=params, grid_num_y=grid_num_y, grid_num_Y=grid_num_Y,
                           y_upper_bound=y_upper_bound, Y_upper_bound=Y_upper_bound)
    data_loader = DataLoader(Y_train, batch_size=len(Y_train), shuffle=False)

    # ── Build test data ──────────────────────────────────────────────────────
    Y_test = test_data_u(params=params, time_period=time_period)
    _, u_theory_test = LQ_theory(y=torch.ones([len(Y_test), 1]), Y=Y_test, params=params)

    # ── Network and optimizer ────────────────────────────────────────────────
    u_theta   = NN_1D(layers=num_layers, dim_hidden=dim_hidden,
                      hidden_activation=hidden_activation, seed=seed)
    optimizer = torch.optim.Adam(u_theta.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(max_epochs):
        for Y_batch in data_loader:
            optimizer.zero_grad()
            residual = euler_residual(params=params, u_theta=u_theta, Y=Y_batch)
            loss     = (residual ** 2).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10000 == 0:
            print(f"epoch = {epoch}, loss_Euler = {loss.item():.2e}")

        if loss.item() < tol:
            print(f"Early stopping at epoch {epoch}, loss_Euler = {loss.item():.2e}")
            break

    # ── Evaluation ───────────────────────────────────────────────────────────
    u_theta.eval()
    with torch.no_grad():
        train_loss = max(loss.item(), tol)

        res_test  = euler_residual(params=params, u_theta=u_theta, Y=Y_test)
        test_loss = max((res_test ** 2).mean().item(), tol)

        u_hat            = u_theta(Y_test)
        abs_rel_err_test = torch.abs((u_hat - u_theory_test) / u_theory_test)

    return {
        'optimal_policy'   : u_theta,
        'param_num'        : count_parameters(u_theta),
        'abs_rel_err_test' : abs_rel_err_test,
        'u_nn'             : u_hat.squeeze().detach().numpy(),
        'u_theory'         : u_theory_test.squeeze().detach().numpy(),
        'Y_test'           : Y_test.squeeze().numpy(),
        'train_loss_euler' : train_loss,
        'test_loss_euler'  : test_loss,
    }
#----------------------------------------------------------------------------

# quantiles 
bottom_quantile = 0.1
top_quantile = 0.9
#--------------------------------------------------------------------

if __name__ == "__main__":
    # Running the experiments
    grid_dim_hidden = np.arange(1, 8)
    results_policy_1_layer_relu = []

    os.makedirs("results/lq", exist_ok=True)
    for h in grid_dim_hidden:
        for seed in range(50):
            print(f"dim_hidden = {h} | seed = {seed}")
            out = training_u(
                params            = param,
                seed              = seed,
                dim_hidden        = h,
                num_layers        = 1,
                hidden_activation = nn.ReLU
            )

            results_policy_1_layer_relu.append({
                'dim_hidden'                                    : h,
                'seed'                                          : seed,
                'param_num_u'                                   : out['param_num'],
                'train_loss_euler'                              : out['train_loss_euler'],
                'test_loss_euler'                               : out['test_loss_euler'],
                'abs_rel_err_test_policy_median'                : out['abs_rel_err_test'].median().item(),
                'abs_rel_err_test_policy_max'                   : out['abs_rel_err_test'].max().item(),
                'abs_rel_err_test_policy_10_percentile'         : out['abs_rel_err_test'].quantile(bottom_quantile).item(),
                'abs_rel_err_test_policy_90_percentile'         : out['abs_rel_err_test'].quantile(top_quantile).item(),
            })

        # Save after every run — guards against kernel crash during long experiment
            pd.DataFrame(results_policy_1_layer_relu).to_csv(
                'results/lq/LQ_results_policy_1_layer_relu.csv', index=False
            )

