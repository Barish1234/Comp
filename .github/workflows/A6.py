#!/usr/bin/env python3
"""
Root finding examples:
1) f1(x) = log(x/2) - sin(5x/2) on [1.5, 3.0] using Bisection and Regula-Falsi
2) f2(x) = -x - cos(x) - find a bracket starting from [2,4] and then refine (bisection)
Tolerance: 1e-6 (adjustable)
"""

import math

# ---------- Functions ----------
def f1(x):
    # f(x) = ln(x/2) - sin(5x/2)
    return math.log(x / 2.0) - math.sin(2.5 * x)

def f2(x):
    # f(x) = -x - cos(x)
    return -x - math.cos(x)

# ---------- Bisection ----------
def bisection(func, a, b, tol=1e-6, maxiter=1000):
    fa = func(a); fb = func(b)
    if fa * fb > 0:
        raise ValueError("Bisection error: f(a) and f(b) have same sign.")
    for i in range(1, maxiter+1):
        c = 0.5 * (a + b)
        fc = func(c)
        # stopping criteria: function value small or interval small
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c, i
        if fa * fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise RuntimeError("Bisection did not converge within maxiter")

# ---------- Regula Falsi (False Position) ----------
def regula_falsi(func, a, b, tol=1e-6, maxiter=10000):
    fa = func(a); fb = func(b)
    if fa * fb > 0:
        raise ValueError("Regula Falsi error: f(a) and f(b) have same sign.")
    c = a
    for i in range(1, maxiter+1):
        # linear interpolation
        c_new = (a * fb - b * fa) / (fb - fa)
        fc = func(c_new)
        # stopping criterion: function small or step small
        if abs(fc) < tol or abs(c_new - c) < tol:
            return c_new, i
        # update interval preserving sign change
        if fa * fc < 0:
            b, fb = c_new, fc
        else:
            a, fa = c_new, fc
        c = c_new
    raise RuntimeError("Regula Falsi did not converge within maxiter")

# ---------- Bracket finder (expand interval) ----------
def find_bracket(func, a, b, expand_factor=1.0, max_expand=50):
    """
    Expand interval [a,b] symmetrically until sign change found or limit hit.
    expand_factor: multiplier of current width to expand each step (1.0 means expand by width each side)
    """
    fa = func(a); fb = func(b)
    if fa * fb <= 0:
        return a, b
    for step in range(max_expand):
        width = (b - a)
        a = a - width * expand_factor
        b = b + width * expand_factor
        fa = func(a); fb = func(b)
        if fa * fb <= 0:
            return a, b
    return None  # failed to find bracket

# ---------- Main / Examples ----------
if __name__ == "__main__":
    tol = 1e-6

    # Problem 1: root of f1 on [1.5, 3.0]
    a1, b1 = 1.5, 3.0
    print("Problem 1: f1(x) = log(x/2) - sin(5x/2) on [1.5, 3.0]")
    print("f(1.5) =", f1(a1), "  f(3.0) =", f1(b1))
    # Bisection
    root_bisect, it_bis = bisection(f1, a1, b1, tol=tol)
    print("Bisection -> root = {:.12f}, iterations = {}, f(root) = {:.3e}".format(root_bisect, it_bis, f1(root_bisect)))
    # Regula Falsi
    root_rf, it_rf = regula_falsi(f1, a1, b1, tol=tol)
    print("Regula Falsi -> root = {:.12f}, iterations = {}, f(root) = {:.3e}".format(root_rf, it_rf, f1(root_rf)))
    print("Absolute difference between roots: {:.3e}".format(abs(root_bisect - root_rf)))
    print()

    # Problem 2: f2(x) = -x - cos(x) ; start with [2,4]
    print("Problem 2: f2(x) = -x - cos(x)")
    a2, b2 = 2.0, 4.0
    print("Initial interval [2,4]: f(2) = {:.6g}, f(4) = {:.6g}".format(f2(a2), f2(b2)))
    bracket = find_bracket(f2, a2, b2, expand_factor=1.0, max_expand=100)
    if bracket is None:
        print("Could not find a bracket by expansion.")
    else:
        A, B = bracket
        print("Found bracket [{:.6g}, {:.6g}] with f(A) = {:.6g}, f(B) = {:.6g}".format(A, B, f2(A), f2(B)))
        # It's often convenient to find a small bracket; try scanning integer pairs inside the found bracket
        # to find an integer subinterval with sign change
        small_bracket = None
        low_int = math.floor(A)
        high_int = math.ceil(B)
        for xi in range(low_int, high_int):
            if xi < low_int or xi+1 > high_int: 
                continue
            if f2(xi) * f2(xi+1) < 0:
                small_bracket = (xi, xi+1)
                break
        if small_bracket is None:
            # fallback: use the expanded bracket
            small_bracket = (A, B)
        a_s, b_s = small_bracket
        print("Using bracket [{:.6g}, {:.6g}] for root refinement.".format(a_s, b_s))
        root2, it2 = bisection(f2, a_s, b_s, tol=1e-12)  # refine to high precision
        print("Root (bisection) = {:.12f}, iterations = {}, f(root) = {:.3e}".format(root2, it2, f2(root2)))

