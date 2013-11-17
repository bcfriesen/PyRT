# read parameters from YAML file
import yaml
stream = open("test.yaml")
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
print(Lambda_star)

# get source function from the formal solution
from source_fn import calc_source_fn
source_fn_n = calc_source_fn(epsilon, J_fs)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

# if Lstar is full then we should be able to do regular Lambda iteration with
# it. this will tell us if we've constructed it correctly.
for i in range(5):
    J_fs = np.dot(Lambda_star, source_fn_n)
    source_fn_n = calc_source_fn(epsilon, J_fs)
    ax.plot(chi_grid, J_fs)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1.0e-2, 1.0e1)
fig.savefig('derp.png')
