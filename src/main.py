import numpy as np
np.set_printoptions(linewidth=200)

# number of angle points
n_mu_pts = 10

# number of physical grid depth points
n_depth_pts = 10

# thermalization parameter. 1 = LTE; 0 = pure scattering
epsilon = 1.0e-3

# "old" source function estimate
source_fn_n = np.zeros(n_depth_pts)

from planck import planck_fn
# initial "guess" for the source function
source_fn_n[:]  = planck_fn(1)

import astropy.units as u
# physical grid
radial_grid = np.linspace(1, 10, n_depth_pts) * u.cm

# opacity grid
chi_grid = np.logspace(-7, 3, n_depth_pts) * (1/u.cm)

# angular grid
mu_grid = np.linspace(-1, 1, n_mu_pts)

rays = []
from ray import ray
for mu in mu_grid:
    rays.append(ray(mu, n_depth_pts, radial_grid))

# let's get some useful (nonzero) values to start
for each_ray in rays:
    each_ray.calc_tau(n_depth_pts, radial_grid, chi_grid)
    each_ray.formal_soln(n_depth_pts, source_fn_n)

from moments import calc_J
J_fs = np.empty(n_depth_pts)
J_fs = calc_J(rays, n_mu_pts, n_depth_pts, mu_grid)

# build tri-diagonal component of Lambda matrix
Lambda_star = np.zeros([n_depth_pts, n_depth_pts])

from calc_Lambda_star import calc_Lambda_star
Lambda_star = calc_Lambda_star(n_depth_pts, n_mu_pts, rays, mu_grid)

# get source function from the formal solution
from source_fn import calc_source_fn
source_fn_n = calc_source_fn(epsilon, J_fs)

planck_grid = np.zeros(n_depth_pts)
for i in range(len(planck_grid)):
    planck_grid[i] = planck_fn(1)

planck_grid = np.zeros(n_depth_pts)
planck_grid[:] = planck_fn(1)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
Delta_S = np.zeros([n_depth_pts])
for i in range(50):
    for each_ray in rays:
        each_ray.formal_soln(n_depth_pts, source_fn_n)
    J_fs = calc_J(rays, n_mu_pts, n_depth_pts, mu_grid)
    source_fn_np1 = np.linalg.solve(1.0 - (1.0 - epsilon) * Lambda_star, (1.0 - epsilon) * (J_fs - np.dot(Lambda_star, source_fn_n)) + epsilon*planck_grid)
    ax.plot(chi_grid, J_fs)
    source_fn_n = source_fn_np1
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1.0e-2, 2.0e0)
fig.savefig('derp.png')
