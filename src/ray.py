import numpy as np
from planck import planck_fn
import astropy.units as u
from grid_functions import *
from math import fabs, exp

class ray:
    def __init__(self, mu, n_depth_pts, radial_grid):
        self.mu = mu
        self.I_lam = np.zeros(n_depth_pts)
        # outward pointing rays should have diffusion as their initial condition
        if self.mu > 0:
            self.I_lam[0] = planck_fn(1) # TODO: make this the diffusion condition
        # our object is not illuminated from some other source so inward pointing rays have zero intensity to start
        else:
            self.I_lam[0] = 0
        self.tau_grid = np.zeros(n_depth_pts)

    def calc_tau(self, n_depth_pts, radial_grid, chi_grid):
        """Given the opacity grid, calculate the optical depth along the ray"""
        self.tau_grid[0] = (0 * u.dimensionless_unscaled)
        for i, depth in enumerate(radial_grid):
            if i > 0:
                grid_idx_im1 = get_grid_index_for_ray_point(self, i-1, n_depth_pts)
                grid_idx_i   = get_grid_index_for_ray_point(self, i  , n_depth_pts)
                self.tau_grid[i] = (self.tau_grid[i-1] + (0.5 * (chi_grid[grid_idx_im1] + chi_grid[grid_idx_i]) * fabs(radial_grid[grid_idx_i] - radial_grid[grid_idx_im1]) / fabs(self.mu)))

    def Delta_tau(self, i):
        return (self.tau_grid[i+1] - self.tau_grid[i])

    # coefficients for source function interpolation
    def alpha(self, i):
        return (1 - exp(-self.Delta_tau(i-1)) - ((self.Delta_tau(i-1) - 1 + exp(-self.Delta_tau(i-1)))/self.Delta_tau(i-1)))
    def beta(self, i):
        return ((self.Delta_tau(i-1) - 1 + exp(-self.Delta_tau(i-1))) / self.Delta_tau(i-1))
    def gamma(self, i):
        return (0)

    # more source function interpolation stuff (see my Eq. 10 for definition of Delta_I)
    def Delta_I(self, i, n_depth_pts, source_fn):
        grid_idx_im1 = get_grid_index_for_ray_point(self, i-1, n_depth_pts)
        grid_idx_i   = get_grid_index_for_ray_point(self, i  , n_depth_pts)
        grid_idx_ip1 = get_grid_index_for_ray_point(self, i+1, n_depth_pts)
        if i < n_depth_pts-1:
            return (self.alpha(i) * source_fn[grid_idx_im1] + self.beta(i) * source_fn[grid_idx_i] + self.gamma(i) * source_fn[grid_idx_ip1])
        elif i == n_depth_pts-1:
            return (self.alpha(i) * source_fn[grid_idx_im1] + self.beta(i) * source_fn[grid_idx_i])

    # perform a formal solution along the ray to get new I
    def formal_soln(self, n_depth_pts, source_fn):
        for i in range(1, n_depth_pts):
            self.I_lam[i] = self.I_lam[i-1] * exp(-self.Delta_tau(i-1)) + self.Delta_I(i, n_depth_pts, source_fn)
