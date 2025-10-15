import numpy as np

def gauss_jordan(A, b, tol=1e-12):
    """
    Solve A x = b using Gauss-Jordan elimination with partial row pivoting.
    A : (n,n) array-like
    b : (n,) or (n,1) array-like
    Returns x as a 1-D numpy array.
    Raises ValueError if the matrix is singular (within tol).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1,1)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square.")
    # Build augmented matrix
    aug = np.hstack((A, b))
    for col in range(n):
        # Partial pivoting: find row with max abs value in this column at or below 'col'
        pivot_row = np.argmax(np.abs(aug[col:, col])) + col
        if abs(aug[pivot_row, col]) < tol:
            raise ValueError(f"Matrix is singular (pivot too small at column {col}).")
        # Swap current row with pivot_row if needed
        if pivot_row != col:
            aug[[col, pivot_row], :] = aug[[pivot_row, col], :]
        # Normalize pivot row
        pivot_val = aug[col, col]
        aug[col, :] = aug[col, :] / pivot_val
        # Eliminate all other rows
        for r in range(n):
            if r == col:
                continue
            factor = aug[r, col]
            if abs(factor) > 0:
                aug[r, :] -= factor * aug[col, :]
    # After reduction, solution is rightmost column
    x = aug[:, -1].copy()
    return x
