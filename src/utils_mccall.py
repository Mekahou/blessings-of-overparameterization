#---------------------------------------------------------------------
# Packages

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# For closed-form solution
from scipy.integrate import quad
from scipy.optimize import fsolve

#--------------------------------------------------------------------
# Parameters
def get_parameters(B=1.0, c=0.1, beta=0.9):
    return {
        'B': B,
        'c': c,
        'beta': beta
    }

#--------------------------------------------------------------------
# Closed-form solution
def v_theory(w, params):
    B = params["B"]
    c = params["c"]
    β = params["beta"]
    ## PDF of wages uniformly distributed on [0, B]
    f = lambda z: 1/B if 0 <= z <= B else 0

    def indifference(w_candidate):
        LHS = w_candidate - c
        integrand = lambda z: (z - w_candidate) * f(z)
        integral, error = quad(integrand, w_candidate, B)
        RHS = (β/(1-β)) * integral
        return LHS - RHS

    w_bar = fsolve(indifference, x0=0.5)[0]
    reject_value = w_bar/(1-β)
    accept_value = w/(1-β)
    index = (w> w_bar)*1.0
    
    return (index*accept_value)+ ((1-index)*reject_value)

# --------------------------------------------------------------------
# Data generation
def grid_w(params, grid_num):
    B = params["B"]
    grid = torch.linspace(0.0, B, steps=grid_num).unsqueeze(dim=1)
    return grid     

# --------------------------------------------------------------------
# Expectation of value function for unemployed workers
def E_v(model,a,b,n):
    # model: the neural network (or any callable)
    # a: lower bound of the unifrom distribution
    # b: upper bound of the uniform distribution
    # n: number of the nodes Gauss-legendre quadrature
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes_tensor = torch.tensor(nodes, dtype=torch.float32).unsqueeze(-1)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
    adjusted_nodes = ((b - a) / 2) * nodes_tensor + ((a + b) / 2)
    integral = ((b - a) / 2) * torch.sum(weights_tensor * model(adjusted_nodes))
    expectation = integral/(b-a)
    return expectation

# ---------------------------------------------------------------------
# Bellman Residuals   
def bellman_residual(params, v_theta, w):
    B = params["B"]
    c = params["c"]
    β = params["beta"]

    lhs_v        = v_theta(w)
    v_employed   = w / (1 - β)
    v_unemployed = c + β * E_v(model=v_theta, a=0, b=B, n=50)
    rhs_v        = torch.maximum(v_employed, v_unemployed)
    return lhs_v - rhs_v              

#---------------------------------------------------------------------                                                                                       
# Neural Networks
class NN(nn.Module):
    def __init__(self,
                 dim_hidden=128,
                 layers=2,
                 hidden_bias=True,
                 hidden_activation=nn.ReLU,
                 output_activation=None,
                 seed=123):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.layers = layers
        self.hidden_bias = hidden_bias
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.seed = seed

        # Set seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        module = []
        
        # First layer
        module.append(nn.Linear(1, self.dim_hidden, bias=self.hidden_bias))
        if self.hidden_activation is not None:
            module.append(self.hidden_activation())

        # Additional hidden layers
        for i in range(self.layers - 1):
            module.append(nn.Linear(self.dim_hidden, self.dim_hidden, bias=self.hidden_bias))
            if self.hidden_activation is not None:
                module.append(self.hidden_activation())

        # Output layer
        module.append(nn.Linear(self.dim_hidden, 1))
        if self.output_activation is not None:
            module.append(self.output_activation())

        self.q = nn.Sequential(*module)

    def forward(self, x):
        out = self.q(x)
        return out
# -----------------------------------------------------------------------

# A function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

