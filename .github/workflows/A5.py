import numpy as np
import itertools

def is_symmetric(A, tol=1e-12):
    A = np.array(A, dtype=float)
    return np.allclose(A, A.T, atol=tol, rtol=0)

def cholesky_decompose(A, tol=1e-12):
    A = np.array(A, dtype=float)
    if not is_symmetric(A, tol=tol):
        raise ValueError("Matrix is not symmetric.")
    n = A.shape[0]
    L = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i,k]*L[j,k] for k in range(j))
            if i==j:
                val = A[i,i] - s
                if val <= tol:
                    raise ValueError(f"Matrix not positive definite at diag {i} (value {val})")
                L[i,i] = (val)**0.5
            else:
                L[i,j] = (A[i,j] - s) / L[j,j]
    return L

def forward_substitution(L, b):
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        s = sum(L[i,j]*y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i,i]
    return y

def backward_substitution(U, y):
    U = np.array(U, dtype=float)
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        s = sum(U[i,j]*x[j] for j in range(i+1, n))
        x[i] = (y[i] - s) / U[i,i]
    return x

def solve_cholesky(A, b):
    L = cholesky_decompose(A)
    y = forward_substitution(L, b)
    x = backward_substitution(L.T, y)
    return x, L

def jacobi(A, b, x0=None, tol=1e-6, maxiter=100000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)
    D = np.diag(A).copy()
    if np.any(np.abs(D) < 1e-15):
        raise ValueError("Zero diagonal entry; cannot apply Jacobi without pivoting.")
    R = A - np.diagflat(D)
    for k in range(1, maxiter+1):
        x_new = (b - R.dot(x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    raise RuntimeError("Jacobi did not converge within maxiter")

def gauss_seidel(A, b, x0=None, tol=1e-6, maxiter=100000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)
    for k in range(1, maxiter+1):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i,j]*x_new[j] for j in range(i))  # using newest values
            s2 = sum(A[i,j]*x[j] for j in range(i+1, n))  # old values
            if abs(A[i,i]) < 1e-15:
                raise ValueError("Zero diagonal entry in A; cannot apply GS without pivoting.")
            x_new[i] = (b[i] - s1 - s2) / A[i,i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    raise RuntimeError("Gauss-Seidel did not converge within maxiter")

def is_diagonally_dominant(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    for i in range(n):
        if abs(A[i,i]) < sum(abs(A[i,j]) for j in range(n) if j!=i):
            return False
    return True

def find_row_permutation_for_diagonal_dominance(A):
    """
    Try all row permutations to find one that makes matrix strictly (or weakly) diagonally dominant by rows.
    Returns permutation (list of row indices) or None if not found.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    idxs = list(range(n))
    for perm in itertools.permutations(idxs):
        P = np.array(A[list(perm), :], dtype=float)
        ok = True
        for i in range(n):
            if abs(P[i,i]) < sum(abs(P[i,j]) for j in range(n) if j!=i):
                ok = False
                break
        if ok:
            return list(perm)
    return None

def permute_system(A, b, perm):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    P = np.array(perm, dtype=int)
    return A[P,:], b[P]
