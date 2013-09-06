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

# build tri-diagonal component of Lambda matrix

# TODO: check boundary conditions (j = 0 and j = n_depth_points). Did I miss anything?

Lambda_star = np.zeros([n_depth_pts, n_depth_pts])

for j, ray in enumerate(rays):
    if j > 0:
        Lambda_star[j-1, j] = ray.gamma(j-1)
    if j < n_depth_pts-2:
        Lambda_star[j+1, j] = Lambda_star[  j, j] * exp(-(ray.Delta_tau(j+1))) + ray.alpha(j+1)
    if j > 0 and j < n_depth_pts-2:
        Lambda_star[  j, j] = Lambda_star[j-1, j] * exp(-(ray.Delta_tau(j))) + ray.beta(j)

# <codecell>


