# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

# <codecell>

# number of angle points
n_mu_pts = 50


# number of optical depth points
n_tau_pts = 10

# <codecell>

# angular grid
mu_grid = np.linspace(-1, 1, n_mu_pts)

# positive angle points
mu_grid_m = mu_grid[:n_mu_pts/2]
# negative angle points
mu_grid_p = mu_grid[n_mu_pts/2 + 1:]

# optical depth grid
tau_grid = np.logspace(-7, 3, n_tau_pts)

# <codecell>

from math import fabs

def delta_tau(i, j):
    return ((tau_grid[i] - tau_grid[i-1]) / fabs(mu_grid[j]))

# <codecell>

# source function interpolation coefficent helpers
from math import exp, pow

def e0(tau_idx, mu_idx):
    return (1 - exp(-delta_tau(tau_idx-1, mu_idx)))
def e1(tau_idx, mu_idx):
    return (delta_tau(tau_idx-1, mu_idx) - e0(tau_idx, mu_idx))
def e2(tau_idx, mu_idx):
    return (pow(delta_tau(tau_idx-1, mu_idx), 2) - 2*e1(tau_idx, mu_idx))

# <codecell>

# source function interpolation coefficents (LINEAR)

def alpha_m(tau_idx, mu_idx):
    return (e0(tau_idx, mu_idx) - e1(tau_idx, mu_idx)/delta_tau(tau_idx-1, mu_idx))
def beta_m(tau_idx, mu_idx):
    return (e1(tau_idx, mu_idx) / delta_tau(tau_idx-1, mu_idx))
def gamma_m(tau_idx, mu_idx):
    return (0)

def alpha_p(tau_idx, mu_idx):
    return (0)
def beta_p(tau_idx, mu_idx):
    return (e1(tau_idx+1, mu_idx) / delta_tau(tau_idx, mu_idx))
def gamma_p(tau_idx, mu_idx):
    return (e0(tau_idx+1, mu_idx) - e1(tau_idx+1, mu_idx)/delta_tau(tau_idx, mu_idx))

# <codecell>

# build tri-diagonal component of Lambda matrix
from scipy.integrate import simps

i_hat_m_im1 = np.empty([n_tau_pts, n_mu_pts])
i_hat_m_i   = np.empty([n_tau_pts, n_mu_pts])
i_hat_m_ip1 = np.empty([n_tau_pts, n_mu_pts])

i_hat_p_im1 = np.empty([n_tau_pts, n_mu_pts])
i_hat_p_i   = np.empty([n_tau_pts, n_mu_pts])
i_hat_p_ip1 = np.empty([n_tau_pts, n_mu_pts])

Lambda = np.zeros([n_tau_pts, n_tau_pts])

for i in range(1, n_tau_pts-1):
    for j in range(n_mu_pts):
        if (i == 1):
            i_hat_m_im1[i, j] = 0
            i_hat_m_i  [i, j] = 0
            i_hat_m_ip1[i, j] = 0
        else:
            i_hat_m_im1[i, j] =  gamma_m(i-1, j)
            i_hat_m_i  [i, j] =  gamma_m(i-1, j) * exp(-delta_tau(i-1, j)) + beta_m(i, j)
            i_hat_m_ip1[i, j] = (gamma_m(i-1, j) * exp(-delta_tau(i-1, j)) + beta_m(i, j)) * exp(-delta_tau(i, j)) + alpha_m(i+1, j)

for i in range(1, n_tau_pts-1):
    for j in range(n_mu_pts):
        if (i == n_tau_pts-1):
            i_hat_p_ip1[i, j] = 0
            i_hat_p_i  [i, j] = 0
            i_hat_p_im1[i, j] = 0
        else:
            i_hat_p_ip1[i, j] =  alpha_p(i+1, j)
            i_hat_p_i  [i, j] =  alpha_p(i+1, j) * exp(-delta_tau(i, j)) + beta_p(i, j)
            i_hat_p_im1[i, j] = (alpha_p(i+1, j) * exp(-delta_tau(i, j)) + beta_p(i, j)) * exp(-delta_tau(i, j)) + gamma_p(i+1, j)

for i in range(1, n_tau_pts-1):
    Lambda[i-1, i] = 0.25 * simps(i_hat_m_im1[i] + i_hat_m_ip1[i])
    Lambda[i  , i] = 0.25 * simps(i_hat_m_i  [i] + i_hat_m_i  [i])
    Lambda[i+1, i] = 0.25 * simps(i_hat_m_ip1[i] + i_hat_m_ip1[i])

# <codecell>


