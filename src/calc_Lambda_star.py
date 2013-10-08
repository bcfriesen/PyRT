from grid_functions import get_grid_index_for_ray_point, get_ray_index_for_grid_point
from math import exp
from scipy.integrate import simps
import numpy as np

def calc_Lambda_star(Lambda_star, n_depth_pts, n_mu_pts, rays, mu_grid):
    inorm_tmp = np.zeros([n_depth_pts, n_mu_pts])
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

    return(Lambda_star)
