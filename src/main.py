# number of angle points
n_mu_pts = 10

# number of physical grid depth points
n_depth_pts = 100

import numpy as np

# Planck function
# TODO: make this actually the Planck function
def planck_fn(T):
    return (1)

# thermalization parameter. 1 = LTE; 0 = pure scattering
epsilon = 1.0e-4

# mean intensity
J_n   = np.zeros(n_depth_pts)
# initial "guess" for J
J_n[:]  = 2

# source function, assumed to be isotropic (so no angle dependence)
source_fn = np.zeros(n_depth_pts)

def calc_source_fn():
    source_fn[:] = epsilon * planck_fn(1) + (1 - epsilon) * J_n[:]
    print("Source function: ", source_fn)

calc_source_fn()

import astropy.units as u

# physical grid
radial_grid = np.linspace(1, 10, n_depth_pts) * u.cm

# opacity grid
chi_grid = np.logspace(-7, 3, n_depth_pts) * (1/u.cm)

def get_grid_index_for_ray_point(ray, ray_idx):
    """Given a ray and a particular point along that ray, return the corresponding grid point index."""
    if ray.mu < 0:
        return (ray_idx)
    else:
        return (n_depth_pts - ray_idx - 1)

def get_ray_index_for_grid_point(ray, grid_idx):
    """Given a ray and a particular point on the physical grid, return the index along that ray corresponding to that point."""
    if ray.mu < 0:
        return (grid_idx)
    else:
        return (n_depth_pts - (grid_idx + 1))


from math import fabs, exp

class ray:
    def __init__(self, mu):
        self.mu = mu
        self.I_lam = np.zeros(n_depth_pts)
        # outward pointing rays should have diffusion as their initial condition
        if self.mu > 0:
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
                grid_idx_im1 = get_grid_index_for_ray_point(self, i-1)
                grid_idx_i   = get_grid_index_for_ray_point(self, i  )
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
    def Delta_I(self, i):
        grid_idx_im1 = get_grid_index_for_ray_point(self, i-1)
        grid_idx_i   = get_grid_index_for_ray_point(self, i  )
        grid_idx_ip1 = get_grid_index_for_ray_point(self, i+1)
        if i < n_depth_pts-1:
            return (self.alpha(i) * source_fn[grid_idx_im1] + self.beta(i) * source_fn[grid_idx_i] + self.gamma(i) * source_fn[grid_idx_ip1])
        elif i == n_depth_pts-1:
            return (self.alpha(i) * source_fn[grid_idx_im1] + self.beta(i) * source_fn[grid_idx_i])

    # perform a formal solution along the ray to get new I
    def formal_soln(self):
        for i in range(1, n_depth_pts):
            self.I_lam[i] = self.I_lam[i-1] * exp(-self.Delta_tau(i-1)) + self.Delta_I(i)

# angular grid
mu_grid = np.linspace(-1, 1, n_mu_pts)

rays = []
for mu in mu_grid:
    rays.append(ray(mu))

# let's get some useful (nonzero) values to start
for ray in rays:
    ray.calc_tau()
    ray.formal_soln()

from scipy.integrate import simps

# build tri-diagonal component of Lambda matrix

Lambda_star = np.zeros([n_depth_pts, n_depth_pts])

inorm_tmp = np.zeros([n_depth_pts, n_mu_pts])

for l in range(1, n_depth_pts-2):
    for j, ray in enumerate(rays):
        # get ray index corresponding to physical grid index l
        ray_idx_l   = get_ray_index_for_grid_point(ray, l)

        # get physical grid indices corresponding to i+1 and i-1 along the ray
        grid_idx_lim1 = get_grid_index_for_ray_point(ray, ray_idx_l-1)
        grid_idx_lip1 = get_grid_index_for_ray_point(ray, ray_idx_l+1)

        inorm_tmp[grid_idx_lim1, j] =  ray.gamma(ray_idx_l-1)
        inorm_tmp[l]                =  ray.gamma(ray_idx_l-1) * exp(-ray.Delta_tau(ray_idx_l-1)) + ray.beta(ray_idx_l)
        inorm_tmp[grid_idx_lip1, j] = (ray.gamma(ray_idx_l-1) * exp(-ray.Delta_tau(ray_idx_l-1)) + ray.beta(ray_idx_l)) * exp(-ray.Delta_tau(ray_idx_l)) + ray.alpha(ray_idx_l+1)

    # TODO: these +1/-1 offsets are hard-coded, works in 1-D plane parallel,
    # but won't work in more complex geometries
    Lambda_star[l-1, l] = 0.5 * simps(inorm_tmp[grid_idx_lim1, :], mu_grid)
    Lambda_star[l  , l] = 0.5 * simps(inorm_tmp[l,             :], mu_grid)
    Lambda_star[l+1, l] = 0.5 * simps(inorm_tmp[grid_idx_lip1, :], mu_grid)

# boundary cases.

# surface
l = 0
inorm_tmp[:, :] = 0
for j, ray in enumerate(rays):
    # at the surface, outgoing rays (those with mu > 0) only have i-1 and i components, no i+1
    if ray.mu > 0:
        ray_idx_l = get_ray_index_for_grid_point(ray, l)

        grid_idx_lim1 = get_grid_index_for_ray_point(ray, ray_idx_l-1)

        inorm_tmp[grid_idx_lim1, j] = ray.gamma(ray_idx_l-1)
        inorm_tmp[l,             j] = ray.gamma(ray_idx_l-1) * exp(-ray.Delta_tau(ray_idx_l-1)) + ray.beta(ray_idx_l)
    else:
        # at the surface we don't have an "i-1" term on incoming rays (those with mu < 0)
        # no illumination from the surface, so incoming rays at the surface have I(surface) = 0
        ray_idx_l = get_ray_index_for_grid_point(ray, l)

        grid_idx_lip1 = get_grid_index_for_ray_point(ray, ray_idx_l+1)

        inorm_tmp[l,             j] = 0
        inorm_tmp[grid_idx_lip1, j] = ray.alpha(ray_idx_l+1)

Lambda_star[l,   l] = 0.5 * simps(inorm_tmp[l,             :], mu_grid)
Lambda_star[l+1, l] = 0.5 * simps(inorm_tmp[grid_idx_lip1, :], mu_grid)

inorm_tmp[:, :] = 0

# depth
l = n_depth_pts-1
for j, ray in enumerate(rays):
    # at depth, we don't have an "i-1" term on rays with mu > 0 because they start at depth
    if (ray.mu < 0):
        ray_idx_l    = get_ray_index_for_grid_point(ray, l)

        grid_idx_lim1 = get_grid_index_for_ray_point(ray, ray_idx_l-1)

        inorm_tmp[grid_idx_lim1, j] = ray.gamma(ray_idx_l-1)
        inorm_tmp[l,             j] = ray.gamma(ray_idx_l-1) * exp(-ray.Delta_tau(ray_idx_l-1)) + ray.beta(ray_idx_l)
    else:
        ray_idx_l    = get_ray_index_for_grid_point(ray, l)

        grid_idx_lip1 = get_grid_index_for_ray_point(ray, ray_idx_l+1)

        inorm_tmp[l,             j] = planck_fn(1)
        inorm_tmp[grid_idx_lip1, j] = planck_fn(1) * exp(-ray.Delta_tau(ray_idx_l)) + ray.alpha(ray_idx_l+1)

Lambda_star[l-1, l] = 0.5 * simps(inorm_tmp[grid_idx_lim1, :], mu_grid)
Lambda_star[l  , l] = 0.5 * simps(inorm_tmp[l,             :], mu_grid)

# mean intensity

# calculate mean intensity
def calc_J(rays):
    I_mu = np.zeros(n_mu_pts)
    J_lam = np.zeros(n_depth_pts)
    for i in range(n_depth_pts):
        for j, ray in enumerate(rays):
            I_mu[j] = ray.I_lam[get_ray_index_for_grid_point(ray, j)]
        J_lam[i] = 0.5 * simps(I_mu, mu_grid)
    return (J_lam)

J_np1 = np.zeros(n_depth_pts)
J_fs  = np.zeros(n_depth_pts)

# J from formal solution (calculated earlier)
J_fs[:] = calc_J(rays)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(radial_grid, J_n)
ax.plot(radial_grid, J_fs)
for i in range(10):
    J_np1 = np.linalg.solve(1 - Lambda_star * (1 - epsilon), J_fs - np.dot(Lambda_star, (1 - epsilon) * J_n))
    J_n = J_np1
    calc_source_fn()
    for ray in rays:
        ray.formal_soln()
    J_fs = calc_J(rays)
    ax.plot(radial_grid, J_n)
    print("Mean intensity:", J_n)
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('derp.png')
