#Packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm

# For closed-form solution
import quantecon as qe
from quantecon import LQ

#--------------------------------------------------------------------
# parameters
def get_parameters(beta=0.9,
                   alpha_0=1.0,
                   alpha_1=1.3,
                   gamma=100.0,
                   h_0=0.05,
                   h_1=0.9,
                   y_0=0.1,
                   Y_0=0.1):
    # h_0 and h_1 are set such that Y_bar = 1
    params = {
        'beta': beta,
        'alpha_0': alpha_0,
        'alpha_1': alpha_1,
        'gamma': gamma,
        'h_0': h_0,
        'h_1': h_1,
        'y_0': y_0,
        'Y_0': Y_0
    }
    return params
#--------------------------------------------------------------------

#Closed form solution for LQ model
def LQ_theory(y, Y, params):
    """
    Compute the closed-form value function and policy for the LQ model.

    Parameters
    ----------
    y      : (n, 1) tensor of individual output levels
    Y      : (n, 1) tensor of aggregate output levels
    params : dict from get_parameters()

    Returns
    -------
    v_result : (n, 1) tensor — closed-form value function
    u_result : (n, 1) tensor — closed-form policy
    """
    beta  = params['beta']
    alpha_0 = params['alpha_0']
    alpha_1 = params['alpha_1']
    gamma = params['gamma']
    h_0   = params['h_0']
    h_1   = params['h_1']

    R = np.matrix([[0.0, -alpha_0/2, 0.0],
                   [-alpha_0/2, 0.0, alpha_1/2],
                   [0.0, alpha_1/2, 0.0]])
    Q = gamma / 2
    A = np.matrix([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [h_0, 0.0, h_1]])
    B = np.matrix([[0.0], [1.0], [0.0]])

    lq = LQ(Q, R, A, B, beta=beta)
    P, F, d = lq.stationary_values()

    P_tens = torch.tensor(P, dtype=torch.float32)
    F_tens = torch.tensor(F, dtype=torch.float32)

    ones      = torch.ones([len(y), 1])
    x_tensor  = torch.cat((ones, y, Y), dim=1).T   # (3, n)

    v_result = -torch.diag(x_tensor.T @ P_tens @ x_tensor).unsqueeze(1)  # (n, 1)
    u_result = (-F_tens @ x_tensor).T                                      # (n, 1)

    return v_result, u_result

#--------------------------------------------------------------------

# Create grid for (y, Y)
def grid_x(params, grid_num_y, grid_num_Y, y_upper_bound, Y_upper_bound):
    y_0 = params['y_0']
    Y_0 = params['Y_0']
    grid_y = torch.linspace(y_0, y_upper_bound, steps=grid_num_y).unsqueeze(dim=1)
    grid_Y = torch.linspace(Y_0, Y_upper_bound, steps=grid_num_Y).unsqueeze(dim=1)
    grid_yY = torch.cartesian_prod(grid_y.squeeze(), grid_Y.squeeze())
    return grid_yY, grid_y, grid_Y

#---------------------------------------------------------------------

# Test data generation for evaluation of the policy
def test_data_u(params, time_period):
    """Generate the deterministic Y_t path for test evaluation of the policy.
    Y_t = h_0 + h_1 * Y_{t-1}
    """
    h_0 = params['h_0']
    h_1 = params['h_1']
    Y_0 = params['Y_0']

    Y_t = torch.zeros(time_period, 1)
    Y_t[0] = Y_0
    for t in range(1, time_period):
        Y_t[t] = h_0 + h_1 * Y_t[t - 1]
    return Y_t
# ---------------------------------------------------------------------

# Test data generation for evaluation of the value function
def test_data_v(params, time_period):
    """Simulate the (y_hat, Y) path using the true LQ policy."""
    y_0 = params['y_0']
    Y_0 = params['Y_0']
    h_0 = params['h_0']
    h_1 = params['h_1']

    beta    = params['beta']
    alpha_0 = params['alpha_0']
    alpha_1 = params['alpha_1']
    gamma   = params['gamma']

    R = np.matrix([[0.0, -alpha_0/2, 0.0],
                   [-alpha_0/2, 0.0, alpha_1/2],
                   [0.0, alpha_1/2, 0.0]])
    Q = gamma / 2
    A = np.matrix([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [h_0, 0.0, h_1]])
    B = np.matrix([[0.0], [1.0], [0.0]])

    lq = LQ(Q, R, A, B, beta=beta)
    P, F, d = lq.stationary_values()

    F_tens = torch.tensor(F, dtype=torch.float32)

    y_hat_t = torch.zeros(time_period, 1)
    Y_t     = torch.zeros(time_period, 1)

    y_hat_t[0] = y_0
    Y_t[0]     = Y_0

    for t in range(1, time_period):
        Y_t[t] = h_0 + h_1 * Y_t[t - 1]

    for t in range(1, time_period):
        x_t = torch.tensor([[1.0], [y_hat_t[t-1].item()], [Y_t[t-1].item()]])
        u_t = -(F_tens @ x_t)
        y_hat_t[t] = y_hat_t[t-1] + u_t[0]

    yY_t = torch.cat((y_hat_t, Y_t), dim=1)
    return yY_t, y_hat_t, Y_t
#---------------------------------------------------------------------
# Euler Residuals
def euler_residual(params, u_theta, Y):
    alpha_0 = params['alpha_0']
    alpha_1 = params['alpha_1']
    beta    = params['beta']
    h_0     = params['h_0']
    h_1     = params['h_1']
    gamma   = params['gamma']

    Y_prime   = h_0 + h_1 * Y
    P_prime   = alpha_0 - alpha_1 * Y_prime
    lhs_euler = gamma * u_theta(Y)
    rhs_euler = beta * (P_prime + gamma * u_theta(Y_prime))
    return rhs_euler - lhs_euler

# Bellman residuals
def bellman_residual(params, u_theta, v_theta, yY):
    alpha_0 = params['alpha_0']
    alpha_1 = params['alpha_1']
    beta    = params['beta']
    h_0     = params['h_0']
    h_1     = params['h_1']
    gamma   = params['gamma']

    y = yY[:, [0]]
    Y = yY[:, [1]]

    with torch.no_grad():
        u = u_theta(Y)

    y_prime  = y + u
    Y_prime  = h_0 + h_1 * Y
    yY_prime = torch.cat((y_prime, Y_prime), dim=1)

    lhs_bellman = v_theta(yY)
    rhs_bellman = -(gamma / 2) * (u ** 2) + (alpha_0 - alpha_1 * Y) * y + beta * v_theta(yY_prime)
    return rhs_bellman - lhs_bellman

#---------------------------------------------------------------------

# Neural Networks 
class NN_1D(nn.Module):
    """One-hidden-layer network for 1D input (Y) — used for the policy function."""
    def __init__(self,
                 dim_hidden=128,
                 layers=1,
                 hidden_bias=True,
                 hidden_activation=nn.ReLU,
                 output_activation=None,
                 seed=123):
        super().__init__()
        self.dim_hidden = dim_hidden
        if seed is not None:
            torch.manual_seed(seed)

        module = []
        module.append(nn.Linear(1, dim_hidden, bias=hidden_bias))
        if hidden_activation is not None:
            module.append(hidden_activation())
        for _ in range(layers - 1):
            module.append(nn.Linear(dim_hidden, dim_hidden, bias=hidden_bias))
            if hidden_activation is not None:
                module.append(hidden_activation())
        module.append(nn.Linear(dim_hidden, 1))
        if output_activation is not None:
            module.append(output_activation())

        self.q = nn.Sequential(*module)

    def forward(self, x):
        return self.q(x)

class NN_2D(nn.Module):
    """One-hidden-layer network for 2D input (y, Y) — used for the value function."""
    def __init__(self,
                 dim_hidden=128,
                 layers=1,
                 hidden_bias=True,
                 hidden_activation=nn.ReLU,
                 output_activation=None,
                 seed=123):
        super().__init__()
        self.dim_hidden = dim_hidden
        if seed is not None:
            torch.manual_seed(seed)

        module = []
        module.append(nn.Linear(2, dim_hidden, bias=hidden_bias))
        module.append(hidden_activation())
        for _ in range(layers - 1):
            module.append(nn.Linear(dim_hidden, dim_hidden, bias=hidden_bias))
            module.append(hidden_activation())
        module.append(nn.Linear(dim_hidden, 1))
        if output_activation is not None:
            module.append(output_activation())

        self.q = nn.Sequential(*module)

    def forward(self, x):
        return self.q(x)
    
#---------------------------------------------------------------------

# A functio that counts the parameters of a neural network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

