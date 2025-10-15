import numpy as np

def cholesky_decompose(A, tol=1e-12):
    """
    Compute the Cholesky decomposition A = L L^T for symmetric positive-definite A.
    Returns lower-triangular L.
    Raises ValueError if not positive-definite within tol.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A must be square.")
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= tol:
                    raise ValueError(f"Matrix not positive-definite at diagonal {i}: {val}")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
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

def backward_substitution(LT, y):
    # LT is upper triangular (L^T)
    U = np.array(LT, dtype=float)
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
    D = np.diag(A)
    if any(np.abs(D) < 1e-15):
        raise ValueError("Zero on diagonal, Jacobi not applicable without pivoting.")
    R = A - np.diagflat(D)
    for k in range(1, maxiter+1):
        x_new = (b - R.dot(x)) / D
        # use infinity norm for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    raise RuntimeError(f"Jacobi did not converge within {maxiter} iterations")
