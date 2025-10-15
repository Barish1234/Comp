#!/usr/bin/env python3
"""
Root-finding comparison.

1) f(x) = 3*x + sin(x) - exp(x/2)  on [-1.5, 1.5]
   Methods: Bisection, Regula Falsi, Newton-Raphson (tol = 1e-6)

2) f(x) = x^2 - 2x - 3  solved by fixed-point iteration:
   g1(x) = sqrt(2*x + 3)   -> converges to x = 3 (start near 3)
   g2(x) = -sqrt(2*x + 3)  -> converges to x = -1 (start near -1)
"""

import math

# ------------------- functions -------------------
def f(x):
    # assumed: f(x) = 3x + sin(x) - e^(x/2)
    return 3.0*x + math.sin(x) - math.exp(x/2.0)

def df(x):
    # derivative: 3 + cos(x) - (1/2) e^(x/2)
    return 3.0 + math.cos(x) - 0.5*math.exp(x/2.0)

# ------------------- Bisection -------------------
def bisection(func, a, b, tol=1e-6, maxiter=1000):
    fa, fb = func(a), func(b)
    if fa*fb > 0:
        raise ValueError("Bisection: f(a) and f(b) must have opposite signs.")
    seq = []
    for it in range(1, maxiter+1):
        c = 0.5*(a + b)
        fc = func(c)
        seq.append(c)
        if abs(fc) < tol or (b - a)/2.0 < tol:
            return c, seq
        if fa*fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise RuntimeError("Bisection did not converge")

# ------------------- Regula Falsi -------------------
def regula_falsi(func, a, b, tol=1e-6, maxiter=10000):
    fa, fb = func(a), func(b)
    if fa*fb > 0:
        raise ValueError("Regula Falsi: f(a) and f(b) must have opposite signs.")
    seq = []
    c_old = None
    for it in range(1, maxiter+1):
        # linear interpolation
        c = (a*fb - b*fa) / (fb - fa)
        fc = func(c)
        seq.append(c)
        if abs(fc) < tol or (c_old is not None and abs(c - c_old) < tol):
            return c, seq
        # update interval preserving sign change
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_old = c
    raise RuntimeError("Regula Falsi did not converge")

# ------------------- Newton-Raphson -------------------
def newton(func, dfunc, x0, tol=1e-6, maxiter=100):
    x = x0
    seq = [x]
    for it in range(1, maxiter+1):
        d = dfunc(x)
        if abs(d) < 1e-14:
            raise RuntimeError("Newton: derivative too small.")
        x_new = x - func(x)/d
        seq.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, seq
        x = x_new
    raise RuntimeError("Newton did not converge")

# ------------------- Fixed-point -------------------
def fixed_point(g, x0, tol=1e-6, maxiter=1000):
    x = x0
    seq = [x]
    for it in range(1, maxiter+1):
        x_new = g(x)
        seq.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, seq
        x = x_new
    raise RuntimeError("Fixed-point did not converge")

# g choices for x^2 - 2x - 3 = 0  (roots 3 and -1)
def g1(x):  # for root 3 (positive)
    return math.sqrt(2.0*x + 3.0)

def g2(x):  # for root -1 (negative)
    return -math.sqrt(2.0*x + 3.0)

# ------------------- Driver & comparison -------------------
if __name__ == "__main__":
    tol = 1e-6
    interval = (-1.5, 1.5)
    a, b = interval
    print("Function: f(x) = 3x + sin(x) - exp(x/2)")
    print("Interval:", interval)
    print("f(a) = {:.6g}, f(b) = {:.6g}".format(f(a), f(b)))

    # Bisection
    root_bis, seq_bis = bisection(f, a, b, tol=tol)
    print("\nBisection: root = {:.12f}, iterations = {}, f(root) = {:.3e}".format(
        root_bis, len(seq_bis), f(root_bis)))

    # Regula Falsi
    root_rf, seq_rf = regula_falsi(f, a, b, tol=tol)
    print("Regula Falsi: root = {:.12f}, iterations = {}, f(root) = {:.3e}".format(
        root_rf, len(seq_rf), f(root_rf)))

    # Newton-Raphson (start at midpoint)
    x0 = 0.0
    root_newt, seq_newt = newton(f, df, x0, tol=tol)
    print("Newton-Raphson: root = {:.12f}, iterations = {}, f(root) = {:.3e}".format(
        root_newt, len(seq_newt), f(root_newt)))

    # Use Newton's result as "reference" for error computation
    ref = root_newt

    # compute per-iteration absolute errors
    err_bis = [abs(x - ref) for x in seq_bis]
    err_rf  = [abs(x - ref) for x in seq_rf]
    err_new = [abs(x - ref) for x in seq_newt]

    # Print a short table of convergence (first few iter)
    print("\nConvergence (first iterations) — abs error to Newton root (reference):")
    print("Bisection (iter, x, err):")
    for i, (x, e) in enumerate(zip(seq_bis[:8], err_bis[:8]), start=1):
        print(f"  {i:2d}: x = {x:.10f}, err = {e:.3e}")
    print("  ... (total iterations {})".format(len(seq_bis)))

    print("\nRegula Falsi (iter, x, err):")
    for i, (x, e) in enumerate(zip(seq_rf[:8], err_rf[:8]), start=1):
        print(f"  {i:2d}: x = {x:.10f}, err = {e:.3e}")
    print("  ... (total iterations {})".format(len(seq_rf)))

    print("\nNewton-Raphson (iter, x, err):")
    for i, (x, e) in enumerate(zip(seq_newt[:8], err_new[:8]), start=1):
        print(f"  {i:2d}: x = {x:.10f}, err = {e:.3e}")
    print("  ... (total iterations {})".format(len(seq_newt)))

    # Summary of iteration counts
    print("\nSummary:")
    print("  Bisection iterations:", len(seq_bis))
    print("  Regula Falsi iterations:", len(seq_rf))
    print("  Newton-Raphson iterations:", len(seq_newt))

    # ----------------- Fixed-point for x^2 - 2x - 3 -----------------
    print("\n\nFixed-point for f(x) = x^2 - 2x - 3")
    print("Root candidates: 3 and -1")

    # root near 3 using g1
    x0 = 2.5
    root_fp1, seq_fp1 = fixed_point(g1, x0, tol=1e-6)
    print("g1 starting at {:.2f} -> root ≈ {:.9f}, iterations = {}".format(x0, root_fp1, len(seq_fp1)))

    # root near -1 using g2
    x0 = -1.0
    root_fp2, seq_fp2 = fixed_point(g2, x0, tol=1e-6)
    print("g2 starting at {:.2f} -> root ≈ {:.9f}, iterations = {}".format(x0, root_fp2, len(seq_fp2)))

    print("\nDone.")

