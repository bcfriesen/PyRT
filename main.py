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
n_mu_pts_m = len(mu_grid_m)

# negative angle points
mu_grid_p = mu_grid[n_mu_pts/2 + 1:]
n_mu_pts_p = len(mu_grid_p)

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

# source function, assumed to be isotropic (so no angle dependence)
def source_fn(i):
    return (planck_fn(1)) # TODO: fix this

# Planck function
def planck_fn(T):
    return (1)

# <codecell>

# specific intensity
I_lam_p = np.zeros([n_tau_pts, n_mu_pts_p])
I_lam_m = np.zeros([n_tau_pts, n_mu_pts_m])

# outward intensity at depth should be the diffusion condition (TODO: FIX THIS. RIGHT NOW IT'S JUST THE PLANCK FUNCTION)
I_lam_p[n_tau_pts-1,:] = planck_fn(1)

# inward intensity at the surface is zero
I_lam_m[0,:] = 0

# <codecell>

# formal solution stuff

def Delta_I_p(i, j):
    return (alpha_p(i, j) * source_fn(i-1) + beta_p(i, j) * source_fn(i) + gamma_p(i, j) * source_fn(i+1))

def Delta_I_m(i, j):
    return (alpha_m(i, j) * source_fn(i-1) + beta_m(i, j) * source_fn(i) + gamma_m(i, j) * source_fn(i+1))

def calc_formal_soln():
    for i in range(n_tau_pts-1, 0, -1):
        for j in range(n_mu_pts_p):
            I_lam_p[i, j] = I_lam_p[i+1, j] * exp(-delta_tau(i, j)) + Delta_I_p(i, j)
    for i in range(n_tau_pts):
        for j in range(1, n_mu_pts_m):
            I_lam_m[i, j] = I_lam_m[i-1, j] * exp(-delta_tau(i-1, j)) + Delta_I_m(i, j)

# <codecell>

calc_formal_soln
print(I_lam_p)

# <codecell>


