# number of angle points
n_mu_pts = 10

# number of physical grid depth points
n_depth_pts = 100

# thermalization parameter. 1 = LTE; 0 = pure scattering
epsilon = 1.0e-4

import numpy as np
# mean intensity
J_n   = np.zeros(n_depth_pts)
# initial "guess" for J
J_n[:]  = 2

# source function, assumed to be isotropic (so no angle dependence)
source_fn = np.zeros(n_depth_pts)

from source_fn import calc_source_fn
source_fn = calc_source_fn(source_fn, epsilon, J_n)

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
    each_ray.formal_soln(n_depth_pts, source_fn)
    print(each_ray.mu, each_ray.I_lam)

# build tri-diagonal component of Lambda matrix
Lambda_star = np.zeros([n_depth_pts, n_depth_pts])

from calc_Lambda_star import calc_Lambda_star
Lambda_star = calc_Lambda_star(Lambda_star, n_depth_pts, n_mu_pts, rays, mu_grid)

# mean intensity from the formal solution
J_fs  = np.zeros(n_depth_pts)

from moments import calc_J

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(radial_grid, J_n)
ax.plot(radial_grid, J_fs)
for i in range(10):
    for each_ray in rays:
        each_ray.formal_soln(n_depth_pts, source_fn)
    J_fs = calc_J(rays, n_mu_pts, n_depth_pts, mu_grid)
    ax.plot(chi_grid, J_fs)
    source_fn = calc_source_fn(source_fn, epsilon, J_fs)
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('derp.png')
