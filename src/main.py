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
    return (planck_fn(1)) # TODO: make this something like epsilon * B + (1 - epsilon) * J

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
        # outward pointing rays should have diffusion as their initial condition
        if mu > 0:
            self.I_lam[0] = planck_fn(1) # TODO: make this the diffusion condition
        # our object is not illuminated from some other source so inward pointing rays have zero intensity to start
        else:
            self.I_lam[0] = 0
        self.tau_grid = np.zeros(n_depth_pts)
        
    def calc_tau(self):
        """Given the opacity grid, calculate the optical depth along the ray"""
        self.tau_grid[0] = (0 * u.dimensionless_unscaled)
        for i, depth in enumerate(radial_grid):
            if i > 0:
                self.tau_grid[i] = (self.tau_grid[i-1] + (0.5 * (chi_grid[i] + chi_grid[i-1]) * (radial_grid[i] - radial_grid[i-1]) / fabs(self.mu)))
                
    def Delta_tau(self, i):
        return (self.tau_grid[i+1] - self.tau_grid[i])
    
    # coefficients for source function interpolation
    def alpha(self, i):
        return (((1 - exp(-self.Delta_tau(i-1)))/self.Delta_tau(i-1))  - exp(-self.Delta_tau(i-1)))
    def beta(self, i):
        return (1 - (1 - exp(-self.Delta_tau(i-1)) / self.Delta_tau(i-1)))
    def gamma(self, i):
        return (0)
    
    # more source function interpolation stuff (see my Eq. 10 for definition of Delta_I)
    def Delta_I(self, i):
        return (self.alpha(i) * source_fn(i-1) + self.beta(i) * source_fn(i) + self.gamma(i) * source_fn(i+1))
    
    # perform a formal solution along the ray to get new I
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

# let's get some useful (nonzero) values to start
for ray in rays:
    ray.calc_tau()
    ray.formal_soln()

# <codecell>

from scipy.integrate import simps

# <codecell>

def get_ray_index_for_grid_point(ray, grid_idx):
    """Given a ray and a particular point on the physical grid, return the index along that ray corresponding to that point."""
    if ray.mu < 0:
        return (grid_idx)
    else:
        return (n_depth_pts - (grid_idx + 1))

# <codecell>

from numpy import matrix

# <codecell>

# build tri-diagonal component of Lambda matrix

# TODO: check boundary conditions (j = 0 and j = n_depth_points). Did I miss anything?
# TODO: come up with a cleaner way to show these indices

Lambda_star = np.zeros([n_depth_pts, n_depth_pts])
inorm_im1 = np.zeros(n_mu_pts)
inorm_i   = np.zeros(n_mu_pts)
inorm_ip1 = np.zeros(n_mu_pts)

for l in range(1, n_depth_pts-2):
    for j, ray in enumerate(rays):
        ray_idx_lm1 = get_ray_index_for_grid_point(ray, l-1)
        ray_idx_l   = get_ray_index_for_grid_point(ray, l  )
        ray_idx_lp1 = get_ray_index_for_grid_point(ray, l+1)
        
        inorm_im1[j] =  ray.gamma(ray_idx_lm1)
        inorm_i  [j] =  ray.gamma(ray_idx_lm1) * exp(-ray.Delta_tau(ray_idx_l)) + ray.beta(ray_idx_l)
        inorm_ip1[j] = (ray.gamma(ray_idx_lm1) * exp(-ray.Delta_tau(ray_idx_l)) + ray.beta(ray_idx_l)) * exp(-ray.Delta_tau(ray_idx_lp1)) + ray.alpha(ray_idx_lp1)
                                                    
    Lambda_star[l-1, l] = 0.5 * simps(inorm_im1, mu_grid) 
    Lambda_star[  l, l] = 0.5 * simps(inorm_i  , mu_grid)
    Lambda_star[l+1, l] = 0.5 * simps(inorm_ip1, mu_grid)
    
# boundary cases.

# the l = 0 boundary case I think is just all zeros so skip it
# TODO: check that the above statement is correct

l = n_depth_pts-2
for j, ray in enumerate(rays):
    ray_idx_lm1 = get_ray_index_for_grid_point(ray, l-1)
    ray_idx_l   = get_ray_index_for_grid_point(ray, l  )

    inorm_im1[j] =  ray.gamma(ray_idx_lm1)
    inorm_i  [j] =  ray.gamma(ray_idx_lm1) * exp(-ray.Delta_tau(ray_idx_l)) + ray.beta(ray_idx_l)
    
    Lambda_star[l-1, l] = 0.5 * simps(inorm_im1, mu_grid) 
    Lambda_star[  l, l] = 0.5 * simps(inorm_i  , mu_grid)
    
l = n_depth_pts-1
for j, ray in enumerate(rays):
    ray_idx_lm1 = get_ray_index_for_grid_point(ray, l-1)

    inorm_im1[j] =  ray.gamma(ray_idx_lm1)
    
    Lambda_star[l-1, l] = 0.5 * simps(inorm_im1, mu_grid) 

# <codecell>

# need to convert to a numpy matrix to do things like find the inverse, etc.
Lambda_star = np.asmatrix(Lambda_star)

# <codecell>

