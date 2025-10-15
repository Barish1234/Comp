"""
Simple library file for computational physics utilities.
Saved: comp_phys_utils.py
Contains: lcg generator and helper wrappers.
"""

def lcg(seed, a=1103515245, c=12345, m=32768, n=1000):
    """Linear Congruential Generator.
    Returns list of n integers in [0, m-1]."""
    vals = [0]*n
    x = seed
    for i in range(n):
        x = (a*x + c) % m
        vals[i] = x
    return vals

def lcg_uniform(seed, a=1103515245, c=12345, m=32768, n=1000):
    """Return n floats in [0,1) using LCG."""
    ints = lcg(seed, a, c, m, n)
    return [v / m for v in ints]
