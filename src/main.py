# read parameters from YAML file
import yaml
stream = open("sample.yaml")
data = yaml.load(stream)

import numpy as np
np.set_printoptions(linewidth=200)

# number of angle points
n_mu_pts = data['n_mu_pts']

# number of physical grid depth points
n_depth_pts = data['n_depth_pts']

# number of points along each ray. for plane-parallel this is the same as the
# number of depth points, but for other geometries it will be different.
n_ray_pts = n_depth_pts

# thermalization parameter. 1 = LTE; 0 = pure scattering
epsilon = data['epsilon']

# "old" source function estimate
source_fn_n = np.zeros(n_depth_pts)

from planck import planck_fn
# initial "guess" for the source function
source_fn_n[:] = planck_fn(1)

# physical grid
radial_grid = np.linspace(data['radius_min'], data['radius_max'], n_depth_pts)

# opacity grid
chi_grid = np.logspace(data['log10_chi_min'], data['log10_chi_max'], n_depth_pts)

# angular grid
mu_grid = np.linspace(-1, 1, n_mu_pts)

rays = []
from ray import ray
for mu in mu_grid:
    rays.append(ray(mu, n_ray_pts, radial_grid))

# let's get some useful (nonzero) values to start
for each_ray in rays:
    each_ray.calc_tau(n_depth_pts, radial_grid, chi_grid)
    each_ray.formal_soln(source_fn_n)

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

from matplotlib import pyplot as plt, gridspec
from math import sqrt

golden_ratio = (1.0 + sqrt(5.0)) / 2.0

fig = plt.figure(figsize=(11.0, 11.0 / golden_ratio), dpi=128)

gs = gridspec.GridSpec(1, 1)

ax = plt.subplot(gs[0, 0])

J_new = np.empty(n_depth_pts)
J_old = np.empty(n_depth_pts)

J_old = J_fs

# Right now L_star is full, so that we can do "traditional" Lambda iteration
# with it if we want (to make sure that we're calculating the matrix elements
# correctly). This also means that basically a single iteration of ALI will
# give us the full, globally converged solution because Lambda_star has all the
# information from all the rays through all the voxels. What we're doing is
# essentially equivalent to the integral solution of the RTE.
for i in range(10):
    source_fn_n = calc_source_fn(epsilon, J_old)
    for each_ray in rays:
        each_ray.calc_tau(n_depth_pts, radial_grid, chi_grid)
        each_ray.formal_soln(source_fn_n)
    J_fs = calc_J(rays, n_mu_pts, n_depth_pts, mu_grid)
    J_new = np.linalg.solve(np.identity(n_depth_pts) - (1.0 - epsilon)*Lambda_star, J_fs - np.dot((1.0 - epsilon)*Lambda_star, J_old))
    J_old = J_new
    ax.plot(chi_grid, J_new)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1.0e-8, 1.0e2)
plt.show()
fig.savefig(data['plot_filename'])
