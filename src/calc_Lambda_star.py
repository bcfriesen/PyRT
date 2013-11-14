from grid_functions import get_ray_index_for_grid_point
from math import exp
from scipy.integrate import simps
import numpy as np

def calc_Lambda_star(n_depth_pts, n_mu_pts, rays, mu_grid):

    Lambda_star = np.zeros([n_depth_pts, n_depth_pts])

    # contributions from each ray to Lstar at a given point on the physical
    # grid; these will be integrated over solid angle such that Lambda[S] = J
    i_hat = np.zeros([n_depth_pts, n_mu_pts])

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
        for ll in range(l, n_depth_pts):
            k = get_ray_index_for_grid_point(ray, ll, n_depth_pts)
            i_hat[ll, j] = ray.Lstar_contrib[k]

    for ll in range(l, n_depth_pts):
        Lambda_star[ll, l] = 0.5 * simps(i_hat[ll, :], mu_grid)


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
            for ll in range(l-1, n_depth_pts):
                k = get_ray_index_for_grid_point(ray, ll, n_depth_pts)
                i_hat[ll, j] = ray.Lstar_contrib[k]

        for ll in range(l-1, n_depth_pts):
            Lambda_star[ll, l] = 0.5 * simps(i_hat[ll, :], mu_grid)


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
        for ll in range(l-1, n_depth_pts):
            k = get_ray_index_for_grid_point(ray, ll, n_depth_pts)
            i_hat[ll, j] = ray.Lstar_contrib[k]

    for ll in range(l-1, n_depth_pts):
        Lambda_star[ll, l] = 0.5 * simps(i_hat[ll, :], mu_grid)

    return(Lambda_star)
