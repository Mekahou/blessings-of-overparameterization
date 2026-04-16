# Packages
import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
#---------------------------------------------------------------------

# Model parameters & steady states 
def get_parameters(beta=0.96, alpha=1.0/3, delta=0.1, rho=0.9, sigma=0.01):
    return {'beta': beta, 'alpha': alpha, 'delta': delta, 'rho': rho, 'sigma': sigma}

def steady_state(params):
    k_ss = ((1.0 / params['beta'] - 1.0 + params['delta']) / params['alpha']) ** (1.0 / (params['alpha'] - 1.0))
    c_ss = k_ss ** params['alpha'] - params['delta'] * k_ss
    return k_ss, c_ss

def state_space_bounds(params, k_width=0.5, z_width=3.0):
    k_ss, _ = steady_state(params)
    z_std = params['sigma'] / np.sqrt(1.0 - params['rho']**2)
    return (1.0 - k_width)*k_ss, (1.0 + k_width)*k_ss, -z_width*z_std, z_width*z_std
#---------------------------------------------------------------------

# Training grid & Gauss-Hermite quadrature
def grid_kz(params, num_k=20, num_z=20):
    k_min, k_max, z_min, z_max = state_space_bounds(params)
    grid_k = torch.linspace(k_min, k_max, num_k)
    grid_z = torch.linspace(z_min, z_max, num_z)
    kz = torch.cartesian_prod(grid_k, grid_z)
    return kz, grid_k, grid_z

def gauss_hermite(M=15):
    zeta_np, w_np = hermgauss(M)
    zeta = torch.tensor(zeta_np, dtype=torch.float32)
    w    = torch.tensor(w_np,    dtype=torch.float32) / np.sqrt(np.pi)
    return zeta, w
#---------------------------------------------------------------------

# Neural network 
class NN_2D(nn.Module):
    def __init__(self, dim_hidden=128, layers=1, hidden_activation=nn.ReLU, softplus_beta=1.0, seed=123):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        module = [nn.Linear(2, dim_hidden), hidden_activation()]
        for _ in range(layers - 1):
            module += [nn.Linear(dim_hidden, dim_hidden), hidden_activation()]
        module += [nn.Linear(dim_hidden, 1), nn.Softplus(beta=softplus_beta)]
        self.q = nn.Sequential(*module)

    def forward(self, x):
        return self.q(x)
#---------------------------------------------------------------------

# A function that counts the parameters of a neural network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#----------------------------------------------------------------------

# Euler equation residual
def euler_residual(params, policy_NN, batch, zeta, w):
    beta, alpha, delta, rho, sigma = params['beta'], params['alpha'], params['delta'], params['rho'], params['sigma']
    M = len(w)
    B = len(batch)

    k = batch[:, [0]]
    z = batch[:, [1]]

    k_prime = policy_NN(batch)
    c       = torch.exp(z) * (k**alpha) + (1 - delta)*k - k_prime

    z_prime_nodes = rho*z + torch.sqrt(torch.tensor(2.0))*sigma*zeta.view(1, M)
    exp_z_prime   = torch.exp(z_prime_nodes)

    f_prime = alpha * (k_prime**(alpha - 1))
    R_prime = exp_z_prime * f_prime + 1 - delta

    k_prime_nodes  = k_prime.unsqueeze(1).expand(-1, M, -1)
    z_prime_nodes3 = z_prime_nodes.unsqueeze(2)
    batch_prime    = torch.cat((k_prime_nodes, z_prime_nodes3), dim=2)
    k_2prime       = policy_NN(batch_prime.reshape(B*M, 2)).view(B, M)

    c_prime = exp_z_prime*(k_prime**alpha) + (1 - delta)*k_prime - k_2prime

    rhs = beta * (R_prime / c_prime) @ w.view(M, 1)
    lhs = 1.0 / c
    return lhs - rhs
#---------------------------------------------------------------------

# Value function Benchmark 
def tauchen(rho, sigma, n_z=51, m=3):
    sigma_z = sigma / np.sqrt(1 - rho**2)
    z_grid  = np.linspace(-m*sigma_z, m*sigma_z, n_z)
    step    = z_grid[1] - z_grid[0]
    P = np.zeros((n_z, n_z))
    for i in range(n_z):
        for j in range(n_z):
            lo = (z_grid[j] - step/2 - rho*z_grid[i]) / sigma
            hi = (z_grid[j] + step/2 - rho*z_grid[i]) / sigma
            if j == 0:
                P[i, j] = norm.cdf(hi)
            elif j == n_z - 1:
                P[i, j] = 1 - norm.cdf(lo)
            else:
                P[i, j] = norm.cdf(hi) - norm.cdf(lo)
    return z_grid, P

def value_function_iteration(params, n_k=201, n_z=51, tol=1e-6, max_iter=1000):
    beta, alpha, delta = params['beta'], params['alpha'], params['delta']
    k_min_v, k_max_v, _, _ = state_space_bounds(params)
    k_grid = np.linspace(k_min_v, k_max_v, n_k)
    z_grid, P = tauchen(params['rho'], params['sigma'], n_z)

    _, c_ss = steady_state(params)
    V = np.full((n_k, n_z), np.log(c_ss) / (1 - beta))
    policy_k = np.zeros((n_k, n_z))

    for iteration in range(max_iter):
        V_new = np.zeros((n_k, n_z))
        for i_k in range(n_k):
            k = k_grid[i_k]
            for i_z in range(n_z):
                z = z_grid[i_z]
                resources = np.exp(z) * k**alpha + (1 - delta)*k
                c_vec = resources - k_grid
                valid = c_vec > 0
                if not valid.any():
                    continue
                EV   = V @ P[i_z, :]
                vals = np.where(valid, np.log(np.where(valid, c_vec, 1)) + beta*EV, -np.inf)
                best = np.argmax(vals)
                V_new[i_k, i_z]    = vals[best]
                policy_k[i_k, i_z] = k_grid[best]
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            print(f'VFI converged in {iteration+1} iterations')
            break

    return policy_k, k_grid, z_grid

def build_vfi_interpolator(params):
    policy_k, k_grid, z_grid = value_function_iteration(params)
    return RegularGridInterpolator(
        (k_grid, z_grid), policy_k, method='linear', bounds_error=False, fill_value=None
    )
#---------------------------------------------------------------------

# Simulation of path of (k, z) using the policy function from VFI, for test data
def simulate_path(params, policy_NN, policy_interp, k0, T=29, mode='no_shocks', seed=1):
    """
    Simulate capital and productivity paths using both NN and VFI policies.

    Parameters
    ----------
    mode : 'no_shocks'   - z_t = 0 for all t (deterministic)
           'with_shocks' - z_t follows AR(1) with random innovations
    seed : int - controls the random shocks (only used when mode='with_shocks')
    """
    rho, sigma = params['rho'], params['sigma']

    if mode == 'no_shocks':
        z_path = np.zeros(T)
    elif mode == 'with_shocks':
        np.random.seed(seed)
        eps = np.random.randn(T)
        z_path = np.zeros(T)
        for t in range(T - 1):
            z_path[t + 1] = rho * z_path[t] + sigma * eps[t]
    else:
        raise ValueError("mode must be 'no_shocks' or 'with_shocks'")

    # --- NN path ---
    k_path_nn = np.zeros(T)
    k_path_nn[0] = k0
    policy_NN.eval()
    with torch.no_grad():
        for t in range(T - 1):
            state = torch.tensor([[k_path_nn[t], z_path[t]]], dtype=torch.float32)
            k_path_nn[t + 1] = policy_NN(state).squeeze().item()

    # --- VFI path ---
    k_path_vfi = np.zeros(T)
    k_path_vfi[0] = k0
    for t in range(T - 1):
        k_path_vfi[t + 1] = policy_interp([[k_path_vfi[t], z_path[t]]])[0]

    return k_path_nn, k_path_vfi, z_path