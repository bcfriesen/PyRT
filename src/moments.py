import numpy as np
from grid_functions import *
from scipy.integrate import simps

# calculate mean intensity
def calc_J(rays, n_mu_pts, n_depth_pts, mu_grid):
    I_mu = np.zeros(n_mu_pts)
    J_lam = np.zeros(n_depth_pts)
    for i in range(n_depth_pts):
        for j, each_ray in enumerate(rays):
            I_mu[j] = each_ray.I_lam[get_ray_index_for_grid_point(each_ray, i, n_depth_pts)]
        J_lam[i] = 0.5 * simps(I_mu, mu_grid)
    return (J_lam)
