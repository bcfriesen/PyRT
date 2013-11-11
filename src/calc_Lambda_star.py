from grid_functions import get_ray_index_for_grid_point
from math import exp
from scipy.integrate import simps
import numpy as np

def calc_Lambda_star(Lambda_star, n_depth_pts, n_mu_pts, rays, mu_grid):

    # contributions from each ray to Lstar at a given point on the physical
    # grid; these will be integrated over solid angle such that Lambda[S] = J
    i_hat_at_l = np.zeros(n_mu_pts)
    # matrix elements at the nearest neighbors of grid point l
    i_hat_at_lp1 = np.zeros(n_mu_pts)
    i_hat_at_lm1 = np.zeros(n_mu_pts)

    # surface

    l = 0

    for ray in rays:
        ray.Lstar_contrib[:] = 0.0
        j = get_ray_index_for_grid_point(ray, l, n_depth_pts)
        # for rays coming in from the surface
        if ray.mu < 0.0:
            for i in range(n_depth_pts):
                # rays coming in from the surface have zero incident intensity, so we have no "i <= j-1" values. instead we just start at i = j
                if i == j:
                    ray.Lstar_contrib[i] = 0.0
                if i == j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1)) + ray.alpha(i)
                elif i > j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1))
        # for rays coming up from depth
        else: # ray.mu > 0.0
            for i in range(n_depth_pts):
                # rays coming in from the surface have zero incident intensity, so we have no "i <= j-1" values. instead we just start at i = j
                if i < j-1:
                    ray.Lstar_contrib[i] = 0.0
                elif i == j-1:
                    ray.Lstar_contrib[i] = ray.gamma(i)
                if i == j:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1)) + ray.beta(i)
                if i == j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1)) + ray.alpha(i)
                elif i > j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1))

    for j, ray in enumerate(rays):
        k = get_ray_index_for_grid_point(ray, l, n_depth_pts)
        i_hat_at_l[j] = ray.Lstar_contrib[k]

    for j, ray in enumerate(rays):
        k = get_ray_index_for_grid_point(ray, l+1, n_depth_pts)
        i_hat_at_lp1[j] = ray.Lstar_contrib[k]

    Lambda_star[l, l] = 0.5 * simps(i_hat_at_l, mu_grid)
    Lambda_star[l+1, l] = 0.5 * simps(i_hat_at_lp1, mu_grid)


    # all the non-boundary points
    for l in range(1, n_depth_pts-1):

        for ray in rays:
            ray.Lstar_contrib[:] = 0.0
            j = get_ray_index_for_grid_point(ray, l, n_depth_pts)
            for i in range(n_depth_pts): # march forward along a ray
                if i < j-1:
                    ray.Lstar_contrib[i] = 0.0
                elif i == j-1:
                    ray.Lstar_contrib[i] = ray.gamma(i)
                elif i == j:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1)) + ray.beta(i)
                elif i == j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1)) + ray.alpha(i)
                elif i > j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1))

        for j, ray in enumerate(rays):
            k = get_ray_index_for_grid_point(ray, l-1, n_depth_pts)
            i_hat_at_lm1[j] = ray.Lstar_contrib[k]

        for j, ray in enumerate(rays):
            k = get_ray_index_for_grid_point(ray, l, n_depth_pts)
            i_hat_at_l[j] = ray.Lstar_contrib[k]

        for j, ray in enumerate(rays):
            k = get_ray_index_for_grid_point(ray, l+1, n_depth_pts)
            i_hat_at_lp1[j] = ray.Lstar_contrib[k]

        Lambda_star[l-1, l] = 0.5 * simps(i_hat_at_lm1, mu_grid)
        Lambda_star[l  , l] = 0.5 * simps(i_hat_at_l  , mu_grid)
        Lambda_star[l+1, l] = 0.5 * simps(i_hat_at_lp1, mu_grid)


    # depth

    l = n_depth_pts-1

    for ray in rays:
        ray.Lstar_contrib[:] = 0.0
        j = get_ray_index_for_grid_point(ray, l, n_depth_pts)
        # for rays coming in from the surface
        if ray.mu < 0.0:
            for i in range(n_depth_pts):
                if i < j-1:
                    ray.Lstar_contrib[i] = 0.0
                elif i == j-1:
                    ray.Lstar_contrib[i] = ray.gamma(i)
                if i == j:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1)) + ray.beta(i)
                if i == j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1)) + ray.alpha(i)
                elif i > j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1))
        # for rays coming up from depth
        else: # ray.mu > 0.0
            for i in range(n_depth_pts):
                if i == j:
                    ray.Lstar_contrib[i] = 0.0
                if i == j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1)) + ray.alpha(i)
                elif i > j+1:
                    ray.Lstar_contrib[i] = ray.Lstar_contrib[i-1] * exp(-ray.Delta_tau(i-1))

    for j, ray in enumerate(rays):
        k = get_ray_index_for_grid_point(ray, l-1, n_depth_pts)
        i_hat_at_lm1[j] = ray.Lstar_contrib[k]

    for j, ray in enumerate(rays):
        k = get_ray_index_for_grid_point(ray, l, n_depth_pts)
        i_hat_at_l[j] = ray.Lstar_contrib[k]

    Lambda_star[l-1, l] = 0.5 * simps(i_hat_at_lm1, mu_grid)
    Lambda_star[l  , l] = 0.5 * simps(i_hat_at_l  , mu_grid)

    return(Lambda_star)
