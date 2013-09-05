# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# number of angle points
n_mu_pts = 10

# number of physical grid depth points
n_depth_pts = 10

# <codecell>

# source function, assumed to be isotropic (so no angle dependence)
def source_fn(i):
    return (planck_fn(1)) # TODO: fix this

# Planck function
def planck_fn(T):
    return (1)

# <codecell>

import numpy as np
import astropy.units as u

# <codecell>

# physical grid
radial_grid = np.linspace(1, 10, n_depth_pts) * u.cm

# opacity grid
chi_grid = np.logspace(-7, 3, n_depth_pts) * (1/u.cm)

# <codecell>

from math import fabs, exp

# <codecell>

class ray:
    def __init__(self, mu):
        self.mu = mu
        self.I_lam = np.zeros(n_depth_pts)
        if mu > 0:
            self.I_lam[0] = planck_fn(1)
        else:
            self.I_lam[0] = 0
        self.tau_grid = np.zeros(n_depth_pts)
    
    def calc_tau(self):
        self.tau_grid[0] = (0 * u.dimensionless_unscaled)
        for i, depth in enumerate(radial_grid):
            if i > 0:
                self.tau_grid[i] = (self.tau_grid[i-1] + (0.5 * (chi_grid[i] + chi_grid[i-1]) * (radial_grid[i] - radial_grid[i-1]) / fabs(self.mu)))
                
    def Delta_tau(self, i):
        return (self.tau_grid[i+1] - self.tau_grid[i])
    
    def alpha(self, i):
        return (((1 - exp(-self.Delta_tau(i-1)))/self.Delta_tau(i-1))  - exp(-self.Delta_tau(i-1)))
    def beta(self, i):
        return (1 - (1 - exp(-self.Delta_tau(i-1)) / self.Delta_tau(i-1)))
    def gamma(self, i):
        return (0)
    
    def Delta_I(self, i):
        return (self.alpha(i) * source_fn(i-1) + self.beta(i) * source_fn(i) + self.gamma(i) * source_fn(i+1))
    
    def formal_soln(self):
        for i, depth in enumerate(self.tau_grid):
            if i > 0:
                self.I_lam[i] = self.I_lam[i-1] * exp(-self.Delta_tau(i-1)) + self.Delta_I(i)

# <codecell>

# angular grid
mu_grid = np.linspace(-1, 1, n_mu_pts)

rays = []
for mu in mu_grid:
    rays.append(ray(mu))

# <codecell>

for ray in rays:
    ray.calc_tau()
    ray.formal_soln()

# <codecell>

for ray in rays:
    print(ray.I_lam)

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


