import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import ast
#--------------------------------------------------------------------

# Setting up figure parameters

fontsize = 16
ticksize = 16
figsize  = (10, 8)
params_fig = {
    'font.family'      : 'serif',
    'figure.figsize'   : figsize,
    'figure.dpi'       : 80,
    'figure.edgecolor' : 'k',
    'font.size'        : fontsize,
    'axes.labelsize'   : fontsize,
    'axes.titlesize'   : fontsize,
    'xtick.labelsize'  : ticksize,
    'ytick.labelsize'  : ticksize,
}
plt.rcParams.update(params_fig)
#--------------------------------------------------------------------

# Quantiles
bottom_quantile = 0.1
top_quantile    = 0.9
#--------------------------------------------------------------------

# Importing the CSV files for test-train and relative error for the RBC model
df = pd.read_csv('./results/rbc/RBC_results_1_layer_relu.csv')

## Aggeregate by n_params: number of parameters
df_median     = df.groupby('n_params').median().reset_index()
df_low        = df.groupby('n_params').quantile(bottom_quantile).reset_index()
df_high       = df.groupby('n_params').quantile(top_quantile).reset_index()

n_params = df_median['n_params']
#--------------------------------------------------------------------

# First plot: Train vs Test Loss (Euler Residuals MSE)

plt.figure()
plt.plot(n_params, df_median['train_loss'], '--', color='r', label='Train Loss Euler: Median')
plt.plot(n_params, df_median['test_loss'],  color='k',       label='Test Loss Euler: Median')
plt.fill_between(
    n_params,
    df_low['test_loss'],
    df_high['test_loss'],
    color='k', alpha=0.2, label='Test Loss Euler: 10–90th Percentiles'
)
plt.xlabel('Number of Parameters')
plt.ylabel('Loss: MSE')
plt.title('RBC Model: Train vs Test Loss (Euler Residuals)')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/RBC_test_train_loss.pdf')
plt.close()
#--------------------------------------------------------------------



# Second plot: Relative Error of the Policy Function
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

ax1.plot(n_params, df_median['abs_rel_err_median'], color='k', label='Median: Test Data')
ax1.fill_between(n_params, df_low['abs_rel_err_median'], df_high['abs_rel_err_median'], color='k', alpha=0.2, label='10–90th Percentiles')
ax1.set_xlabel('Number of Parameters')
ax1.set_ylabel(r'$\tilde{\varepsilon}_k$', fontsize=24)
ax1.set_title('RBC Model: Median Abs Relative Error')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc='upper right')

ax2.plot(n_params, df_median['abs_rel_err_max'], color='k', label='Median: Test Data')
ax2.fill_between(n_params, df_low['abs_rel_err_max'], df_high['abs_rel_err_max'], color='k', alpha=0.2, label='10–90th Percentiles')
ax2.set_xlabel('Number of Parameters')
ax2.set_ylabel(r'$\bar{\varepsilon}_k$', fontsize=24)
ax2.set_title('RBC Model: Max Abs Relative Error')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.tick_params(labelleft=True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/RBC_rel_error_median_vs_max.pdf')
plt.close()
#--------------------------------------------------------------------

# Third plot: under vs overparametrized (dim_hidden = 32 vs dim_hidden = 512)
df_512 = pd.read_csv('results/rbc/RBC_results_1_layer_relu_dim_hidden_512.csv')
df_32= pd.read_csv('results/rbc/RBC_results_1_layer_relu_dim_hidden_32.csv')

k_paths_nn_512 = np.array([ast.literal_eval(row) for row in df_512['k_path_nn']])  # (50, T)
k_paths_nn_32  = np.array([ast.literal_eval(row) for row in df_32['k_path_nn']])   # (50, T)
k_path_vfi     = np.array(ast.literal_eval(df_512['k_path_vfi'].iloc[0]))           # (T,)
t              = np.arange(len(k_path_vfi))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Left: H = 32 (no zoom) ───────────────────────────────────────────────
ax = axes[0]
ax.plot(t, k_path_vfi, color='b', linestyle='--', label='VFI')
ax.plot(t, np.median(k_paths_nn_32, axis=0), color='k', label='NN: Median')
ax.fill_between(
    t,
    np.percentile(k_paths_nn_32, 10, axis=0),
    np.percentile(k_paths_nn_32, 90, axis=0),
    color='k', alpha=0.3, label='NN: 10–90th Percentiles'
)
ax.set_xlabel('Time($t$)')
ax.set_ylabel('Capital($k_t$)')
ax.set_title('RBC Model: Capital Path, VFI vs NN ($H$ = 32)')
ax.legend(loc='lower right')

# ── Right: H = 512 (with zoom) ───────────────────────────────────────────
ax = axes[1]
ax.plot(t, k_path_vfi, color='b', linestyle='--', label='VFI')
ax.plot(t, np.median(k_paths_nn_512, axis=0), color='k', label='NN: Median')
ax.fill_between(
    t,
    np.percentile(k_paths_nn_512, 10, axis=0),
    np.percentile(k_paths_nn_512, 90, axis=0),
    color='k', alpha=0.3, label='NN: 10–90th Percentiles'
)
ax.set_xlabel('Time($t$)')
ax.set_ylabel('Capital($k_t$)')
ax.set_title('RBC Model: Capital Path, VFI vs NN ($H$ = 512)')
ax.legend(loc='lower right')

# ── Zoomed inset ──────────────────────────────────────────────────────────
axins = zoomed_inset_axes(
    ax,
    zoom=2.5,
    loc='upper left',
    bbox_to_anchor=(0.4, 0.7, 0, 0),
    bbox_transform=ax.transAxes,
)
axins.plot(t, k_path_vfi, color='b', linestyle='--')
axins.plot(t, np.median(k_paths_nn_512, axis=0), color='k')
axins.fill_between(
    t,
    np.percentile(k_paths_nn_512, 10, axis=0),
    np.percentile(k_paths_nn_512, 90, axis=0),
    color='k', alpha=0.3
)
axins.set_xlim(15, 21)
axins.set_ylim(3.35, 3.60)
axins.tick_params(labelsize=8)
mark_inset(ax, axins, loc1=2, loc2=4, linewidth=0.7, ls='--', ec='0.5')

plt.tight_layout()
plt.savefig('figures/RBC_capital_path_32_vs_512.pdf')
plt.close()
#----------------------------------------------------------------------------

# Fourth plot: Test vs Train Data
from src.utils_rbc import (get_parameters,
                           steady_state,
                           grid_kz, NN_2D, build_vfi_interpolator, simulate_path)

params       = get_parameters()
k_ss, c_ss   = steady_state(params)

# Training data: 20×20 Cartesian grid over (k, z)
kz_train, _, _ = grid_kz(params, num_k=20, num_z=20)
k_train = kz_train[:, 0].numpy()
z_train = kz_train[:, 1].numpy()

# Test data: T=29 VFI-simulated path, with_shocks, seed=1, k0=0.5*k_ss
policy_interp = build_vfi_interpolator(params)
dummy_nn      = NN_2D()
_, k_path_vfi, z_path = simulate_path(
    params, dummy_nn, policy_interp, k0=0.5 * k_ss, T=29, mode='with_shocks', seed=1
)

plt.figure()
plt.scatter(k_train, z_train, color='red', s=15, alpha=0.3, label='Train Data', zorder=2)
plt.scatter(k_path_vfi, z_path, s=30, marker='+', color='blue', label='Test Data', zorder=3)

plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$z$', fontsize=20)
plt.title('RBC Model: Train vs Test Data')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend(loc='lower right', fontsize=22)
plt.savefig('figures/RBC_test_vs_train_data.pdf')
plt.close()


#-------------------------#Robustness Plots #--------------------------------
## Importing the CSV files 
df_relu      = pd.read_csv('results/rbc/RBC_results_2_layer_relu.csv')
df_leakyrelu = pd.read_csv('results/rbc/RBC_results_2_layer_leakyrelu.csv')
df_sigmoid   = pd.read_csv('results/rbc/RBC_results_2_layer_sigmoid.csv')

# ── Aggregate: ReLU ───────────────────────────────────────────────────────────
relu_med  = df_relu.groupby('n_params').median().reset_index()
relu_low  = df_relu.groupby('n_params').quantile(bottom_quantile).reset_index()
relu_high = df_relu.groupby('n_params').quantile(top_quantile).reset_index()
x_relu    = relu_med['n_params']

# ── Aggregate: LeakyReLU ──────────────────────────────────────────────────────
leakyrelu_med  = df_leakyrelu.groupby('n_params').median().reset_index()
leakyrelu_low  = df_leakyrelu.groupby('n_params').quantile(bottom_quantile).reset_index()
leakyrelu_high = df_leakyrelu.groupby('n_params').quantile(top_quantile).reset_index()
x_leakyrelu    = leakyrelu_med['n_params']

# ── Aggregate: Sigmoid ────────────────────────────────────────────────────────
sigmoid_med  = df_sigmoid.groupby('n_params').median().reset_index()
sigmoid_low  = df_sigmoid.groupby('n_params').quantile(bottom_quantile).reset_index()
sigmoid_high = df_sigmoid.groupby('n_params').quantile(top_quantile).reset_index()
x_sigmoid    = sigmoid_med['n_params']

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey='row')

# ── Row 1: Train vs Test Loss ─────────────────────────────────────────────────

# Col 0: ReLU
axes[0, 0].plot(x_relu, relu_med['train_loss'], '--', color='r', label='Train Loss: Median')
axes[0, 0].plot(x_relu, relu_med['test_loss'],        color='k', label='Test Loss: Median')
axes[0, 0].fill_between(x_relu, relu_low['test_loss'], relu_high['test_loss'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[0, 0].set_title('RBC Model: Train vs Test Loss\n(ReLU, 2 layers)')
axes[0, 0].set_xlabel('Number of Parameters')
axes[0, 0].set_ylabel('Loss: MSE')
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].tick_params(labelleft=True)
axes[0, 0].legend(loc='upper right', fontsize=10)

# Col 1: LeakyReLU
axes[0, 1].plot(x_leakyrelu, leakyrelu_med['train_loss'], '--', color='r', label='Train Loss: Median')
axes[0, 1].plot(x_leakyrelu, leakyrelu_med['test_loss'],        color='k', label='Test Loss: Median')
axes[0, 1].fill_between(x_leakyrelu, leakyrelu_low['test_loss'], leakyrelu_high['test_loss'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[0, 1].set_title('RBC Model: Train vs Test Loss\n(Leaky ReLU, 2 layers)')
axes[0, 1].set_xlabel('Number of Parameters')
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].tick_params(labelleft=True)
axes[0, 1].legend(loc='upper right', fontsize=10)

# Col 2: Sigmoid
axes[0, 2].plot(x_sigmoid, sigmoid_med['train_loss'], '--', color='r', label='Train Loss: Median')
axes[0, 2].plot(x_sigmoid, sigmoid_med['test_loss'],        color='k', label='Test Loss: Median')
axes[0, 2].fill_between(x_sigmoid, sigmoid_low['test_loss'], sigmoid_high['test_loss'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[0, 2].set_title('RBC Model: Train vs Test Loss\n(Sigmoid, 2 layers)')
axes[0, 2].set_xlabel('Number of Parameters')
axes[0, 2].set_xscale('log')
axes[0, 2].set_yscale('log')
axes[0, 2].tick_params(labelleft=True)
axes[0, 2].legend(loc='upper right', fontsize=10)

# ── Row 2: Median Abs Relative Error ─────────────────────────────────────────

# Col 0: ReLU
axes[1, 0].plot(x_relu, relu_med['abs_rel_err_median'], color='k', label='Median: Test Data')
axes[1, 0].fill_between(x_relu, relu_low['abs_rel_err_median'], relu_high['abs_rel_err_median'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[1, 0].set_title('Median Abs Relative Error\n(ReLU, 2 layers)')
axes[1, 0].set_xlabel('Number of Parameters')
axes[1, 0].set_ylabel(r'$\tilde{\varepsilon}_k$', fontsize=20)
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].tick_params(labelleft=True)
axes[1, 0].legend(loc='upper right', fontsize=10)

# Col 1: LeakyReLU
axes[1, 1].plot(x_leakyrelu, leakyrelu_med['abs_rel_err_median'], color='k', label='Median: Test Data')
axes[1, 1].fill_between(x_leakyrelu, leakyrelu_low['abs_rel_err_median'], leakyrelu_high['abs_rel_err_median'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[1, 1].set_title('Median Abs Relative Error\n(Leaky ReLU, 2 layers)')
axes[1, 1].set_xlabel('Number of Parameters')
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].tick_params(labelleft=True)
axes[1, 1].legend(loc='upper right', fontsize=10)

# Col 2: Sigmoid
axes[1, 2].plot(x_sigmoid, sigmoid_med['abs_rel_err_median'], color='k', label='Median: Test Data')
axes[1, 2].fill_between(x_sigmoid, sigmoid_low['abs_rel_err_median'], sigmoid_high['abs_rel_err_median'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[1, 2].set_title('Median Abs Relative Error\n(Sigmoid, 2 layers)')
axes[1, 2].set_xlabel('Number of Parameters')
axes[1, 2].set_xscale('log')
axes[1, 2].set_yscale('log')
axes[1, 2].tick_params(labelleft=True)
axes[1, 2].legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/RBC_robustness_2_layers_activations.pdf')
plt.close()