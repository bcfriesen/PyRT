from planck import planck_fn

def calc_source_fn(source_fn, epsilon, J_n):
    source_fn[:] = epsilon * planck_fn(1) + (1 - epsilon) * J_n[:]
