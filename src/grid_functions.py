def get_grid_index_for_ray_point(ray, ray_idx, n_depth_pts):
    """Given a ray and a particular point along that ray, return the corresponding grid point index."""
    if ray.mu < 0:
        return (ray_idx)
    else:
        return (n_depth_pts - ray_idx - 1)

def get_ray_index_for_grid_point(ray, grid_idx, n_depth_pts):
    """Given a ray and a particular point on the physical grid, return the index along that ray corresponding to that point."""
    if ray.mu < 0:
        return (grid_idx)
    else:
        return (n_depth_pts - (grid_idx + 1))
