"""Measured-rate integer allocation; no device work at import."""
import math

def allocate(n, cpu_cost, gpu_cost):
    if not all(math.isfinite(x) and x > 0 for x in (cpu_cost, gpu_cost)):
        raise ValueError('positive measured per-target selection costs required')
    x = n * gpu_cost / (cpu_cost + gpu_cost)
    quota = min({math.floor(x), math.ceil(x)}, key=lambda q: (max(q*cpu_cost, (n-q)*gpu_cost), q))
    return [((i+1)*quota)//n > i*quota//n for i in range(n)]
