from planck import planck_fn
import numpy as np

def calc_source_fn(epsilon, J_n):
    source_fn = np.zeros(len(J_n))
    source_fn[:] = epsilon * planck_fn(1) + (1 - epsilon) * J_n[:]
    return (source_fn)
