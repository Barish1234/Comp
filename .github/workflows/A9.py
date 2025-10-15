import cmath
import math
import random
import numpy as np
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

def poly_eval(coefs, x):
    """Evaluate polynomial and its first two derivatives at x.
    coefs: list/array of coefficients [a_n, a_{n-1}, ..., a_0]
    returns (p, p1, p2)
    """
    n = len(coefs) - 1
    p = coefs[0]
    dp = 0.0
    ddp = 0.0
    for i in range(1, n+1):
        ddp = dp + x * ddp
        dp = p + x * dp
        p = coefs[i] + x * p
    # After Horner-style loop, p is value, dp is p', ddp is p''
    return p, dp, ddp

def laguerre(coefs, x0, maxiters=200, tol=1e-12):
    """Laguerre's method for one root. Returns complex root estimate."""
    n = len(coefs) - 1
    x = complex(x0)
    for i in range(maxiters):
        p, dp, ddp = poly_eval(coefs, x)
        if abs(p) < tol:
            return x
        G = dp / p
        H = G*G - ddp / p
        denom_term = complex((n-1)*(n*H - G*G))
        # safeguard for negative small roundoff making sqrt argument tiny negative
        sqrt_term = cmath.sqrt(denom_term)
        # choose larger denominator in magnitude
        a1 = G + sqrt_term
        a2 = G - sqrt_term
        if abs(a1) > abs(a2):
            a = n / a1
        else:
            a = n / a2
        x_new = x - a
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x  # return last iterate if not converged

def deflate(coefs, root, tol=1e-9):
    """Deflate polynomial by dividing by (x - root). Returns quotient coefficients and remainder."""
    n = len(coefs) - 1
    q = [coefs[0]]  # quotient first coeff
    for i in range(1, n):
        q.append(coefs[i] + q[-1] * root)
    remainder = coefs[-1] + q[-1] * root
    return q, remainder

def find_all_roots(coefs, tol=1e-10):
    """Find all roots using Laguerre + deflation. Returns list of roots (complex)."""
    coefs = [complex(c) for c in coefs]  # ensure complex
    roots = []
    working = coefs.copy()
    while len(working) - 1 > 0:
        n = len(working) - 1
        # choose a starting guess: random complex near unit circle or last coefficient magnitude
        # try several starts if Laguerre doesn't converge to a stable root
        found = None
        for attempt in range(12):
            # heuristic start: random near some scale
            scale = 1.0 + random.random()*2.0
            angle = random.random() * 2*math.pi
            x0 = scale * (math.cos(angle) + 1j*math.sin(angle))
            root = laguerre(working, x0, maxiters=300, tol=tol)
            # polish: one Newton iteration with full polynomial
            p, dp, ddp = poly_eval(working, root)
            if abs(dp) > 1e-16:
                root = root - p/dp
            # check residual
            p_val, _, _ = poly_eval(working, root)
            if abs(p_val) < 1e-6:  # acceptable root for deflation
                found = root
                break
        if found is None:
            # last resort: try real starting points (use spaced seeds)
            for x0 in np.linspace(-5, 5, 21):
                root = laguerre(working, x0, maxiters=400, tol=tol)
                p_val, _, _ = poly_eval(working, root)
                if abs(p_val) < 1e-6:
                    found = root
                    break
        if found is None:
            # If still not found, return what we have (graceful fallback)
            break
        # polish root with a few Newton iterations on the current polynomial
        for _ in range(6):
            p, dp, _ = poly_eval(working, found)
            if abs(dp) < 1e-16:
                break
            found = found - p/dp
        # deflate
        quotient, rem = deflate(working, found)
        # if remainder small, accept and continue
        if abs(rem) > 1e-4:
            # try conjugate if root complex to reduce remainder (sometimes pairing needed)
            # but proceed anyway and warn
            # We'll accept but print a warning later.
            pass
        roots.append(found)
        working = quotient
    return roots

def sort_and_format_roots(roots, tol_real=1e-8):
    # Sort by real part then imag part and format
    roots_sorted = sorted(roots, key=lambda z: (round(z.real,10), round(z.imag,10)))
    out = []
    for r in roots_sorted:
        is_real = abs(r.imag) < tol_real
        val = complex(r.real, 0.0) if is_real else r
        out.append((val, is_real))
    return out

def analyze_polynomial(coefs, name="P"):
    # Find roots
    roots = find_all_roots(coefs)
    formatted = sort_and_format_roots(roots)
    # compute residuals on original polynomial (numpy polyval)
    residues = []
    for r, _ in formatted:
        res = np.polyval(coefs, r)
        residues.append(res)
    # prepare dataframe
    df = pd.DataFrame({
        "root": [complex(r) for r,_ in formatted],
        "is_real (approx)": [is_real for _,is_real in formatted],
        "residual (poly at root)": residues
    })
    print(f"\n{name}: coefficients (highest -> lowest): {coefs}")
    display_dataframe_to_user(f"{name} roots", df)
    # Also print a concise textual summary
    for r, is_real in formatted:
        if is_real:
            print(f"  Real root ≈ {r.real:.12g}   (residual {np.polyval(coefs, r):.2e})")
        else:
            print(f"  Complex root ≈ {r.real:.12g} {'+' if r.imag>=0 else '-'} {abs(r.imag):.12g}j   (residual {np.polyval(coefs, r):.2e})")

# Now run the routine on the three polynomials given in the problem.
P1 = [1.0, -1.0, -7.0, 1.0, 6.0]       # x^4 - x^3 - 7x^2 + x + 6
P2 = [1.0, 0.0, -5.0, 0.0, 4.0]        # x^4 - 5x^2 + 4
P3 = [2.0, 0.0, -19.5, 0.5, 13.5, -4.5] # 2x^5 - 19.5x^3 + 0.5x^2 + 13.5x - 4.5

analyze_polynomial(P1, "P1 (x^4 - x^3 - 7x^2 + x + 6)")
analyze_polynomial(P2, "P2 (x^4 - 5x^2 + 4)")
analyze_polynomial(P3, "P3 (2x^5 - 19.5x^3 + 0.5x^2 + 13.5x - 4.5)")

# Additionally, print numpy.roots for cross-check
print("\nCross-check using numpy.roots:")
for name, coefs in [("P1", P1), ("P2", P2), ("P3", P3)]:
    nr = np.roots(coefs)
    print(f"{name} numpy.roots:")
    for r in nr:
        if abs(r.imag) < 1e-10:
            print(f"  {r.real:.12g} (real)")
        else:
            print(f"  {r.real:.12g} {'+' if r.imag>=0 else '-'} {abs(r.imag):.12g}j")


