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
calc_source_fn(source_fn, epsilon, J_n)

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


# build tri-diagonal component of Lambda matrix
Lambda_star = np.zeros([n_depth_pts, n_depth_pts])
inorm_tmp = np.zeros([n_depth_pts, n_mu_pts])
from grid_functions import get_grid_index_for_ray_point, get_ray_index_for_grid_point
from math import exp
from scipy.integrate import simps
for l in range(1, n_depth_pts-2):
    for j, each_ray in enumerate(rays):
        # get ray index corresponding to physical grid index l
        ray_idx_l   = get_ray_index_for_grid_point(each_ray, l, n_depth_pts)

        # get physical grid indices corresponding to i+1 and i-1 along the ray
        grid_idx_lim1 = get_grid_index_for_ray_point(each_ray, ray_idx_l-1, n_depth_pts)
        grid_idx_lip1 = get_grid_index_for_ray_point(each_ray, ray_idx_l+1, n_depth_pts)

        inorm_tmp[grid_idx_lim1, j] =  each_ray.gamma(ray_idx_l-1)
        inorm_tmp[l]                =  each_ray.gamma(ray_idx_l-1) * exp(-each_ray.Delta_tau(ray_idx_l-1)) + each_ray.beta(ray_idx_l)
        inorm_tmp[grid_idx_lip1, j] = (each_ray.gamma(ray_idx_l-1) * exp(-each_ray.Delta_tau(ray_idx_l-1)) + each_ray.beta(ray_idx_l)) * exp(-each_ray.Delta_tau(ray_idx_l)) + each_ray.alpha(ray_idx_l+1)

    # TODO: these +1/-1 offsets are hard-coded, works in 1-D plane parallel,
    # but won't work in more complex geometries
    Lambda_star[l-1, l] = 0.5 * simps(inorm_tmp[grid_idx_lim1, :], mu_grid)
    Lambda_star[l  , l] = 0.5 * simps(inorm_tmp[l,             :], mu_grid)
    Lambda_star[l+1, l] = 0.5 * simps(inorm_tmp[grid_idx_lip1, :], mu_grid)

# boundary cases.

# surface
l = 0
inorm_tmp[:, :] = 0
for j, each_ray in enumerate(rays):
    # at the surface, outgoing rays (those with mu > 0) only have i-1 and i components, no i+1
    if each_ray.mu > 0:
        ray_idx_l = get_ray_index_for_grid_point(each_ray, l, n_depth_pts)

        grid_idx_lim1 = get_grid_index_for_ray_point(each_ray, ray_idx_l-1, n_depth_pts)

        inorm_tmp[grid_idx_lim1, j] = each_ray.gamma(ray_idx_l-1)
        inorm_tmp[l,             j] = each_ray.gamma(ray_idx_l-1) * exp(-each_ray.Delta_tau(ray_idx_l-1)) + each_ray.beta(ray_idx_l)
    else:
        # at the surface we don't have an "i-1" term on incoming rays (those with mu < 0)
        # no illumination from the surface, so incoming rays at the surface have I(surface) = 0
        ray_idx_l = get_ray_index_for_grid_point(each_ray, l, n_depth_pts)

        grid_idx_lip1 = get_grid_index_for_ray_point(each_ray, ray_idx_l+1, n_depth_pts)

        inorm_tmp[l,             j] = 0
        inorm_tmp[grid_idx_lip1, j] = each_ray.alpha(ray_idx_l+1)

Lambda_star[l,   l] = 0.5 * simps(inorm_tmp[l,             :], mu_grid)
Lambda_star[l+1, l] = 0.5 * simps(inorm_tmp[grid_idx_lip1, :], mu_grid)

inorm_tmp[:, :] = 0

# depth
from planck import planck_fn
l = n_depth_pts-1
for j, each_ray in enumerate(rays):
    # at depth, we don't have an "i-1" term on rays with mu > 0 because they start at depth
    if (each_ray.mu < 0):
        ray_idx_l    = get_ray_index_for_grid_point(each_ray, l, n_depth_pts)

        grid_idx_lim1 = get_grid_index_for_ray_point(each_ray, ray_idx_l-1, n_depth_pts)

        inorm_tmp[grid_idx_lim1, j] = each_ray.gamma(ray_idx_l-1)
        inorm_tmp[l,             j] = each_ray.gamma(ray_idx_l-1) * exp(-each_ray.Delta_tau(ray_idx_l-1)) + each_ray.beta(ray_idx_l)
    else:
        ray_idx_l    = get_ray_index_for_grid_point(each_ray, l, n_depth_pts)

        grid_idx_lip1 = get_grid_index_for_ray_point(each_ray, ray_idx_l+1, n_depth_pts)

        inorm_tmp[l,             j] = planck_fn(1)
        inorm_tmp[grid_idx_lip1, j] = planck_fn(1) * exp(-each_ray.Delta_tau(ray_idx_l)) + each_ray.alpha(ray_idx_l+1)

Lambda_star[l-1, l] = 0.5 * simps(inorm_tmp[grid_idx_lim1, :], mu_grid)
Lambda_star[l  , l] = 0.5 * simps(inorm_tmp[l,             :], mu_grid)

# mean intensity

J_np1 = np.zeros(n_depth_pts)
J_fs  = np.zeros(n_depth_pts)

# J from formal solution (calculated earlier)
from moments import calc_J
J_fs[:] = calc_J(rays, n_mu_pts, n_depth_pts, mu_grid)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(radial_grid, J_n)
ax.plot(radial_grid, J_fs)
for i in range(10):
    J_np1 = np.linalg.solve(1 - Lambda_star * (1 - epsilon), J_fs - np.dot(Lambda_star, (1 - epsilon) * J_n))
    J_n = J_np1
    calc_source_fn(source_fn, epsilon, J_n)
    for each_ray in rays:
        each_ray.formal_soln(n_depth_pts, source_fn)
    J_fs = calc_J(rays, n_mu_pts, n_depth_pts, mu_grid)
    ax.plot(radial_grid, J_n)
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('derp.png')
