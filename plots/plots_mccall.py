# Packages 
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

# Importing the CSV files for test-train and relative error for the McCall model
df = pd.read_csv('./results/mccall/McCall_results_1_layer_relu.csv')

## Aggeregate by param_num_v
df_median = df.groupby('param_num_v').median().reset_index()
df_low    = df.groupby('param_num_v').quantile(bottom_quantile).reset_index()
df_high   = df.groupby('param_num_v').quantile(top_quantile).reset_index()

n_params = df_median['param_num_v']
#--------------------------------------------------------------------

# First plot: Train vs Test Loss (Bellman MSE)

plt.figure()
plt.plot(n_params, df_median['train_loss_bellman'], '--', color='r', label='Train Loss Bellman: Median')
plt.plot(n_params, df_median['test_loss_bellman'],        color='k', label='Test Loss Bellman: Median')
plt.fill_between(
    n_params,
    df_low['test_loss_bellman'],
    df_high['test_loss_bellman'],
    color='k', alpha=0.2, label='Test Loss Bellman: 10–90th Percentiles'
)
plt.xlabel('Number of Parameters')
plt.ylabel('Loss: MSE')
plt.title('McCall Model: Train vs Test Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/Mccall_test_train_loss.pdf')
plt.close()
#--------------------------------------------------------------------



# Second plot: Relative Error of the Value Function
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

ax1.plot(n_params, df_median['abs_rel_err_test_value_function_median'], color='k', label='Median: Test Data')
ax1.fill_between(n_params, df_low['abs_rel_err_test_value_function_median'], df_high['abs_rel_err_test_value_function_median'], color='k', alpha=0.2, label='10–90th Percentiles')
ax1.set_xlabel('Number of Parameters')
ax1.set_ylabel(r'$\tilde{\varepsilon}_v$', fontsize=24)
ax1.set_title('McCall Model: Median Abs Relative Error')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc='upper right')

ax2.plot(n_params, df_median['abs_rel_err_test_value_function_max'], color='k', label='Median: Test Data')
ax2.fill_between(n_params, df_low['abs_rel_err_test_value_function_max'], df_high['abs_rel_err_test_value_function_max'], color='k', alpha=0.2, label='10–90th Percentiles')
ax2.set_xlabel('Number of Parameters')
ax2.set_ylabel(r'$\bar{\varepsilon}_v$', fontsize=24)
ax2.set_title('McCall Model: Max Abs Relative Error')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.tick_params(labelleft=True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/Mccall_rel_error_median_vs_max.pdf')
plt.close()
#--------------------------------------------------------------------

# Third plot: under vs overparametrized (dim_hidden = 24 vs dim_hidden = 512)

## Importing the CSV files
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils_mccall import get_parameters, v_theory , grid_w
import torch

df_512 = pd.read_csv('results/mccall/McCall_results_1_layer_relu_dim_hidden_512.csv')
df_24  = pd.read_csv('results/mccall/McCall_results_1_layer_relu_dim_hidden_24.csv')

v_nns_512 = np.array([ast.literal_eval(row) for row in df_512['v_nn']])   # (50, T)
v_nns_24  = np.array([ast.literal_eval(row) for row in df_24['v_nn']])    # (50, T)
w_test    = np.array(ast.literal_eval(df_512['w_test'].iloc[0]))            # (T,)

# Recompute v_theory on a fine grid for a sharp kink (CSV only has 100 pts)
param_plt    = get_parameters()
w_fine       = grid_w(params=param_plt, grid_num=1000)          # tensor (1000, 1)
w_plt        = w_fine.squeeze().numpy()
v_theory_plt = v_theory(w=w_fine, params=param_plt).squeeze().numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Left: H = 24 (with zoom inset) ───────────────────────────────────────
ax = axes[0]
ax.plot(w_plt, v_theory_plt, color='b', linestyle='--', label='Closed-form Solution')
ax.plot(w_test, np.median(v_nns_24, axis=0), color='k', label='NN: Median')
ax.fill_between(
    w_test,
    np.percentile(v_nns_24, 10, axis=0),
    np.percentile(v_nns_24, 90, axis=0),
    color='k', alpha=0.3, label='NN: 10–90th Percentiles'
)
ax.set_xlabel(r'Wage ($w$)')
ax.set_ylabel(r'Value ($v(w)$)')
ax.set_title('McCall Model: Closed-form vs NN ($H$ = 24)')
ax.legend(loc='upper right', fontsize=13)

axins = zoomed_inset_axes(ax, zoom=6, loc='upper left',
                          bbox_to_anchor=(0.05, 0.98, 0, 0),
                          bbox_transform=ax.transAxes)
axins.plot(w_plt, v_theory_plt, color='b', linestyle='--')
axins.plot(w_test, np.median(v_nns_24, axis=0), color='k')
axins.fill_between(
    w_test,
    np.percentile(v_nns_24, 10, axis=0),
    np.percentile(v_nns_24, 90, axis=0),
    color='k', alpha=0.3
)
axins.set_xlim(0.6, 0.67)
axins.set_ylim(6.48, 6.75)
axins.tick_params(labelbottom=False, labelleft=False)
mark_inset(ax, axins, loc1=2, loc2=4, linewidth=0.7, ls='--', ec='0.5')

# ── Right: H = 512 (with zoom inset) ─────────────────────────────────────
ax = axes[1]
ax.plot(w_plt, v_theory_plt, color='b', linestyle='--', label='Closed-form Solution')
ax.plot(w_test, np.median(v_nns_512, axis=0), color='k', label='NN: Median')
ax.fill_between(
    w_test,
    np.percentile(v_nns_512, 10, axis=0),
    np.percentile(v_nns_512, 90, axis=0),
    color='k', alpha=0.3, label='NN: 10–90th Percentiles'
)
ax.set_xlabel(r'Wage ($w$)')
ax.set_ylabel(r'Value ($v(w)$)')
ax.set_title('McCall Model: Closed-form vs NN ($H$ = 512)')
ax.legend(loc='upper right', fontsize=13)

axins = zoomed_inset_axes(ax, zoom=6, loc='upper left',
                          bbox_to_anchor=(0.05, 0.98, 0, 0),
                          bbox_transform=ax.transAxes)
axins.plot(w_plt, v_theory_plt, color='b', linestyle='--')
axins.plot(w_test, np.median(v_nns_512, axis=0), color='k')
axins.fill_between(
    w_test,
    np.percentile(v_nns_512, 10, axis=0),
    np.percentile(v_nns_512, 90, axis=0),
    color='k', alpha=0.3
)
axins.set_xlim(0.6, 0.67)
axins.set_ylim(6.48, 6.75)
axins.tick_params(labelbottom=False, labelleft=False)
mark_inset(ax, axins, loc1=2, loc2=4, linewidth=0.7, ls='--', ec='0.5')
plt.savefig('figures/McCall_value_function_24_vs_512.pdf')
plt.close()

#-------------------------#Robustness Plots #--------------------------------

## ── Load robustness CSVs ──────────────────────────────────────────────────────
df_relu      = pd.read_csv('results/mccall/McCall_results_2_layer_relu.csv')
df_leakyrelu = pd.read_csv('results/mccall/McCall_results_2_layer_leakyrelu.csv')

## ── Aggregate: ReLU ───────────────────────────────────────────────────────────
relu_med  = df_relu.groupby('param_num_v').median().reset_index()
relu_low  = df_relu.groupby('param_num_v').quantile(bottom_quantile).reset_index()
relu_high = df_relu.groupby('param_num_v').quantile(top_quantile).reset_index()
x_relu    = relu_med['param_num_v']

## ── Aggregate: LeakyReLU ──────────────────────────────────────────────────────
leakyrelu_med  = df_leakyrelu.groupby('param_num_v').median().reset_index()
leakyrelu_low  = df_leakyrelu.groupby('param_num_v').quantile(bottom_quantile).reset_index()
leakyrelu_high = df_leakyrelu.groupby('param_num_v').quantile(top_quantile).reset_index()
x_leakyrelu    = leakyrelu_med['param_num_v']

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')

# ── Row 1: Train vs Test Loss ─────────────────────────────────────────────────

# Col 0: ReLU
axes[0, 0].plot(x_relu, relu_med['train_loss_bellman'], '--', color='r', label='Train Loss: Median')
axes[0, 0].plot(x_relu, relu_med['test_loss_bellman'],        color='k', label='Test Loss: Median')
axes[0, 0].fill_between(x_relu, relu_low['test_loss_bellman'], relu_high['test_loss_bellman'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[0, 0].set_title('McCall Model: Train vs Test Loss (ReLU, 2 layers)')
axes[0, 0].set_xlabel('Number of Parameters')
axes[0, 0].set_ylabel('Loss: MSE')
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].tick_params(labelleft=True)
axes[0, 0].legend(loc='upper right', fontsize=10)

# Col 1: LeakyReLU
axes[0, 1].plot(x_leakyrelu, leakyrelu_med['train_loss_bellman'], '--', color='r', label='Train Loss: Median')
axes[0, 1].plot(x_leakyrelu, leakyrelu_med['test_loss_bellman'],        color='k', label='Test Loss: Median')
axes[0, 1].fill_between(x_leakyrelu, leakyrelu_low['test_loss_bellman'], leakyrelu_high['test_loss_bellman'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[0, 1].set_title('McCall Model: Train vs Test Loss (Leaky ReLU, 2 layers)')
axes[0, 1].set_xlabel('Number of Parameters')
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].tick_params(labelleft=True)
axes[0, 1].legend(loc='upper right', fontsize=10)

# ── Row 2: Median Absolute Relative Error ────────────────────────────────────

# Col 0: ReLU
axes[1, 0].plot(x_relu, relu_med['abs_rel_err_test_value_function_median'], color='k', label='Median: Test Data')
axes[1, 0].fill_between(x_relu, relu_low['abs_rel_err_test_value_function_median'], relu_high['abs_rel_err_test_value_function_median'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[1, 0].set_title('Median Abs Relative Error (ReLU, 2 layers)')
axes[1, 0].set_xlabel('Number of Parameters')
axes[1, 0].set_ylabel(r'$\tilde{\varepsilon}_v$', fontsize=20)
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].tick_params(labelleft=True)
axes[1, 0].legend(loc='upper right', fontsize=10)

# Col 1: LeakyReLU
axes[1, 1].plot(x_leakyrelu, leakyrelu_med['abs_rel_err_test_value_function_median'], color='k', label='Median: Test Data')
axes[1, 1].fill_between(x_leakyrelu, leakyrelu_low['abs_rel_err_test_value_function_median'], leakyrelu_high['abs_rel_err_test_value_function_median'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[1, 1].set_title('Median Abs Relative Error (Leaky ReLU, 2 layers)')
axes[1, 1].set_xlabel('Number of Parameters')
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].tick_params(labelleft=True)
axes[1, 1].legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/McCall_robustness_2_layers_activations.pdf')
plt.close()