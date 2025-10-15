mport numpy as np

def lu_doolittle(A, tol=1e-12):
    """
    Doolittle LU decomposition: A = L U
    L has 1s on the diagonal, U is upper triangular.
    Returns (L, U).
    Raises ValueError if the matrix appears singular (zero pivot).
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A must be square.")
    L = np.eye(n, dtype=float)
    U = np.zeros((n,n), dtype=float)
    for k in range(n):
        # Compute U[k, k..n-1]
        for j in range(k, n):
            U[k,j] = A[k,j] - sum(L[k,s]*U[s,j] for s in range(k))
        if abs(U[k,k]) < tol:
            raise ValueError(f"Zero (or near-zero) pivot encountered at U[{k},{k}]")
        # Compute L[k+1..n-1, k]
        for i in range(k+1, n):
            L[i,k] = (A[i,k] - sum(L[i,s]*U[s,k] for s in range(k))) / U[k,k]
    return L, U

def forward_substitution(L, b):
    """Solve L y = b where L is lower-triangular with unit diagonal (or general)."""
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        s = sum(L[i,j]*y[j] for j in range(i))
        if abs(L[i,i]) < 1e-15:
            raise ValueError(f"Zero diagonal in L at {i}")
        y[i] = (b[i] - s) / L[i,i]
    return y

def backward_substitution(U, y):
    """Solve U x = y where U is upper-triangular."""
    U = np.array(U, dtype=float)
    y = np.array(y, dtype=float)
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        s = sum(U[i,j]*x[j] for j in range(i+1, n))
        if abs(U[i,i]) < 1e-15:
            raise ValueError(f"Zero diagonal in U at {i}")
        x[i] = (y[i] - s) / U[i,i]
    return x

def solve_via_lu(A, b):
    """Solve A x = b using Doolittle LU and forward/back substitution.
    Returns solution x, and (L, U).
    """
    L, U = lu_doolittle(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x, L, U
