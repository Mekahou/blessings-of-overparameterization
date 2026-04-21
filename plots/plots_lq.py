# Packages 
import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import ast
#--------------------------------------------------

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

#------------------------------------------------LQ Policy plots------------------------------------------

# First plot: LQ policy test vs train losses

# Importing the CSV files for test-train and relative error for the LQ policy
df = pd.read_csv('./results/lq/LQ_results_policy_1_layer_relu.csv')

df_median = df.groupby('param_num_u').median().reset_index()
df_low    = df.groupby('param_num_u').quantile(bottom_quantile).reset_index()
df_high   = df.groupby('param_num_u').quantile(top_quantile).reset_index()

n_params = df_median['param_num_u']

plt.figure()
plt.plot(n_params, df_median['train_loss_euler'], '--', color='r', label='Train Loss Euler: Median')
plt.plot(n_params, df_median['test_loss_euler'],        color='k', label='Test Loss Euler: Median')
plt.fill_between(
    n_params,
    df_low['test_loss_euler'],
    df_high['test_loss_euler'],
    color='k', alpha=0.2, label='Test Loss Euler: 10–90th Percentiles'
)
plt.xlabel('Number of Parameters')
plt.ylabel('Loss: MSE')
plt.title('LQ Model: Train vs Test Loss (Euler Residuals)')
plt.yscale('log')
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/LQ_policy_test_train_loss.pdf')
plt.close()

# Second plot: LQ policy relative errors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

ax1.plot(n_params, df_median['abs_rel_err_test_policy_median'], color='k', label='Median: Test Data')
ax1.fill_between(n_params, df_low['abs_rel_err_test_policy_median'], df_high['abs_rel_err_test_policy_median'], color='k', alpha=0.2, label='10–90th Percentiles')
ax1.set_xlabel('Number of Parameters')
ax1.set_ylabel(r'$\tilde{\varepsilon}_u$', fontsize=24)
ax1.set_title('LQ Model: Policy, Median Abs Relative Error')
ax1.set_yscale('log')
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.legend(loc='upper right')

ax2.plot(n_params, df_median['abs_rel_err_test_policy_max'], color='k', label='Median: Test Data')
ax2.fill_between(n_params, df_low['abs_rel_err_test_policy_max'], df_high['abs_rel_err_test_policy_max'], color='k', alpha=0.2, label='10–90th Percentiles')
ax2.set_xlabel('Number of Parameters')
ax2.set_ylabel(r'$\bar{\varepsilon}_u$', fontsize=24)
ax2.set_title('LQ Model: Policy, Max Abs Relative Error')
ax2.set_yscale('log')
ax2.tick_params(labelleft=True)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/LQ_policy_rel_error_median_vs_max.pdf')
plt.close()

# Third plot: under vs overparametrized (dim_hidden = 3 vs dim_hidden = 32)
df_32    = pd.read_csv('results/lq/LQ_results_policy_1_layer_relu_dim_hidden_32.csv')
df_3     = pd.read_csv('results/lq/LQ_results_policy_1_layer_relu_dim_hidden_3.csv')
u_nns_32 = np.array([ast.literal_eval(row) for row in df_32['u_nn']])   # (50, T)
u_nns_3  = np.array([ast.literal_eval(row) for row in df_3['u_nn']])    # (50, T)
u_theory = np.array(ast.literal_eval(df_32['u_theory'].iloc[0]))         # (T,)
t        = np.arange(len(u_theory))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, u_nns, H in zip(axes, [u_nns_3, u_nns_32], [3, 32]):
    ax.plot(t, u_theory, color='b', linestyle='--', label='Closed-form Solution')
    ax.plot(t, np.median(u_nns, axis=0), color='k', label='NN: Median')
    ax.fill_between(
        t,
        np.percentile(u_nns, 10, axis=0),
        np.percentile(u_nns, 90, axis=0),
        color='k', alpha=0.3, label='NN: 10–90th Percentiles'
    )
    ax.set_xlabel('Time($t$)')
    ax.set_ylabel(r'Policy ($u_t$)')
    ax.set_title(f'LQ Model: Policy Path, Closed-form vs NN ($H$ = {H})')
    ax.legend(loc='upper right')

    # ── Zoomed inset (over-parameterized only) ────────────────────────────
    if H == 32:
        x1, x2 = 0, 3
        axins = ax.inset_axes([0.5, 0.4, 0.45, 0.35])
        axins.plot(t, u_theory, color='b', linestyle='--')
        axins.plot(t, np.median(u_nns, axis=0), color='k')
        axins.fill_between(
            t,
            np.percentile(u_nns, 10, axis=0),
            np.percentile(u_nns, 90, axis=0),
            color='k', alpha=0.3
        )
        mask = t <= x2
        y_lo = min(np.percentile(u_nns[:, mask], 10), u_theory[mask].min())
        y_hi = max(np.percentile(u_nns[:, mask], 90), u_theory[mask].max())
        pad  = 0.02 * (y_hi - y_lo)
        axins.set_xlim(x1, x2)
        axins.set_ylim(y_lo - pad, y_hi + pad)
        axins.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        axins.tick_params(labelsize=8)
        ax.indicate_inset_zoom(axins, edgecolor='gray')

plt.tight_layout()
plt.savefig('figures/LQ_policy_path_3_vs_32.pdf')
plt.close()

# Fourth plot: Robustness (Sigmoid and Leaky ReLU activations)

df_sigmoid   = pd.read_csv('results/lq/LQ_results_policy_1_layer_sigmoid.csv')
df_leakyrelu = pd.read_csv('results/lq/LQ_results_policy_1_layer_leakyrelu.csv')

df_sigmoid_median   = df_sigmoid.groupby('param_num_u').median().reset_index()
df_sigmoid_low      = df_sigmoid.groupby('param_num_u').quantile(bottom_quantile).reset_index()
df_sigmoid_high     = df_sigmoid.groupby('param_num_u').quantile(top_quantile).reset_index()

df_leakyrelu_median = df_leakyrelu.groupby('param_num_u').median().reset_index()
df_leakyrelu_low    = df_leakyrelu.groupby('param_num_u').quantile(bottom_quantile).reset_index()
df_leakyrelu_high   = df_leakyrelu.groupby('param_num_u').quantile(top_quantile).reset_index()

x_sigmoid   = df_sigmoid_median['param_num_u']
x_leakyrelu = df_leakyrelu_median['param_num_u']

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')

# ── Sigmoid: Train/Test Loss ──────────────────────────────────────────────────
axes[0, 0].plot(x_sigmoid, df_sigmoid_median['train_loss_euler'], '--', color='r', label='Train Loss: Median')
axes[0, 0].plot(x_sigmoid, df_sigmoid_median['test_loss_euler'],        color='k', label='Test Loss: Median')
axes[0, 0].fill_between(x_sigmoid, df_sigmoid_low['test_loss_euler'], df_sigmoid_high['test_loss_euler'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[0, 0].set_title('LQ Model: Policy, Train vs Test Loss (Sigmoid)')
axes[0, 0].set_xlabel('Number of Parameters')
axes[0, 0].set_ylabel('Loss: MSE')
axes[0, 0].set_yscale('log')
axes[0, 0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
axes[0, 0].tick_params(labelleft=True)
axes[0, 0].legend(loc='upper right', fontsize=10)

# ── LeakyReLU: Train/Test Loss ────────────────────────────────────────────────
axes[0, 1].plot(x_leakyrelu, df_leakyrelu_median['train_loss_euler'], '--', color='r', label='Train Loss: Median')
axes[0, 1].plot(x_leakyrelu, df_leakyrelu_median['test_loss_euler'],        color='k', label='Test Loss: Median')
axes[0, 1].fill_between(x_leakyrelu, df_leakyrelu_low['test_loss_euler'], df_leakyrelu_high['test_loss_euler'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[0, 1].set_title('LQ Model: Policy, Train vs Test Loss (Leaky ReLU)')
axes[0, 1].set_xlabel('Number of Parameters')
axes[0, 1].set_yscale('log')
axes[0, 1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
axes[0, 1].tick_params(labelleft=True)
axes[0, 1].legend(loc='upper right', fontsize=10)

# ── Sigmoid: Policy Error ─────────────────────────────────────────────────────
axes[1, 0].plot(x_sigmoid, df_sigmoid_median['abs_rel_err_test_policy_median'], color='k', label='Median: Test Data')
axes[1, 0].fill_between(x_sigmoid, df_sigmoid_low['abs_rel_err_test_policy_median'], df_sigmoid_high['abs_rel_err_test_policy_median'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[1, 0].set_title('Median Abs Relative Error (Sigmoid)')
axes[1, 0].set_xlabel('Number of Parameters')
axes[1, 0].set_ylabel(r'$\tilde{\varepsilon}_u$', fontsize=20)
axes[1, 0].set_yscale('log')
axes[1, 0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
axes[1, 0].tick_params(labelleft=True)
axes[1, 0].legend(loc='upper right', fontsize=10)

# ── LeakyReLU: Policy Error ───────────────────────────────────────────────────
axes[1, 1].plot(x_leakyrelu, df_leakyrelu_median['abs_rel_err_test_policy_median'], color='k', label='Median: Test Data')
axes[1, 1].fill_between(x_leakyrelu, df_leakyrelu_low['abs_rel_err_test_policy_median'], df_leakyrelu_high['abs_rel_err_test_policy_median'], color='k', alpha=0.2, label='10–90th Percentiles')
axes[1, 1].set_title('Median Abs Relative Error (Leaky ReLU)')
axes[1, 1].set_xlabel('Number of Parameters')
axes[1, 1].set_yscale('log')
axes[1, 1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
axes[1, 1].tick_params(labelleft=True)
axes[1, 1].legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/LQ_policy_robustness_activations.pdf')
plt.close()

#------------------------------------------------LQ Value Function plots------------------------------------------

# Fifth plot: LQ Value function test vs train losses
df_v = pd.read_csv('results/lq/LQ_results_value_1_layer_relu.csv')

## ── Aggregate by param_num ────────────────────────────────────────────────
df_v_median = df_v.groupby('param_num').median().reset_index()
df_v_low    = df_v.groupby('param_num').quantile(bottom_quantile).reset_index()
df_v_high   = df_v.groupby('param_num').quantile(top_quantile).reset_index()

n_params_v = df_v_median['param_num']

plt.figure()
plt.plot(n_params_v, df_v_median['train_loss_bellman'], '--', color='r', label='Train Loss Bellman: Median')
plt.plot(n_params_v, df_v_median['test_loss_bellman'],        color='k', label='Test Loss Bellman: Median')
plt.fill_between(
    n_params_v,
    df_v_low['test_loss_bellman'],
    df_v_high['test_loss_bellman'],
    color='k', alpha=0.2, label='Test Loss Bellman: 10–90th Percentiles'
)
plt.xlabel('Number of Parameters')
plt.ylabel('Loss: MSE')
plt.title('LQ Model: Train vs Test Loss (Bellman Residuals)')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/LQ_value_test_train_loss.pdf')
plt.close()


# Sixth plot:  LQ Value Function relative errors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

ax1.plot(n_params_v, df_v_median['abs_rel_err_test_median'], color='k', label='Median: Test Data')
ax1.fill_between(n_params_v, df_v_low['abs_rel_err_test_median'], df_v_high['abs_rel_err_test_median'], color='k', alpha=0.2, label='10–90th Percentiles')
ax1.set_xlabel('Number of Parameters')
ax1.set_ylabel(r'$\tilde{\varepsilon}_v$', fontsize=24)
ax1.set_title('LQ Model: Value Function, Median Abs Relative Error')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc='upper right')

ax2.plot(n_params_v, df_v_median['abs_rel_err_test_max'], color='k', label='Median: Test Data')
ax2.fill_between(n_params_v, df_v_low['abs_rel_err_test_max'], df_v_high['abs_rel_err_test_max'], color='k', alpha=0.2, label='10–90th Percentiles')
ax2.set_xlabel('Number of Parameters')
ax2.set_ylabel(r'$\bar{\varepsilon}_v$', fontsize=24)
ax2.set_title('LQ Model: Value Function, Max Abs Relative Error')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.tick_params(labelleft=True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/LQ_value_rel_error_median_vs_max.pdf')
plt.close()

