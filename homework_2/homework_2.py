'''
    Homework 2
'''



import numpy as np

def lu_decomposition_inplace(A, dU, eps):
    """
    Compute LU decomposition of A in-place.
    A is modified so that:
      - The lower-triangular part (including diagonal) stores L.
      - The strictly upper-triangular part stores U (off-diagonals),
        while U's diagonal is stored separately in dU.
    The factorization satisfies A = L * U.
    
    Parameters:
        A   : (n,n) numpy array (will be modified)
        dU  : (n,) numpy array, given diagonal values for U (with |dU[i]|>=eps)
        eps : tolerance to check for small denominators.
    
    Returns:
        None (A is modified in place)
    """
    n = A.shape[0]
    for p in range(n):
        # Compute L[p, j] for j=0,...,p-1
        for j in range(p):
            sum_LU = 0.0
            for k in range(j):
                sum_LU += A[p, k] * A[k, j]  # L[p,k] in A[p,k] and U[k,j] in A[k,j]
            # U[j,j] = dU[j]
            if abs(dU[j]) < eps:
                raise ValueError(f"Zero pivot encountered in U at index {j}")
            A[p, j] = (A[p, j] - sum_LU) / dU[j]
        # Diagonal element for L: j = p
        sum_LU = 0.0
        for k in range(p):
            sum_LU += A[p, k] * A[k, p]
        A[p, p] = A[p, p] - sum_LU
        if abs(A[p, p]) < eps:
            raise ValueError(f"Zero pivot encountered in L at index {p}")
        # Set U[p,p] = dU[p] (this value is kept externally; we do not store it in A)
        # Now compute U[p, j] for j = p+1,...,n-1 (store in A[p,j])
        for j in range(p+1, n):
            sum_LU = 0.0
            for k in range(p):
                sum_LU += A[p, k] * A[k, j]
            A[p, j] = (A[p, j] - sum_LU) / A[p, p]
    # At this point, A contains L in the lower part (diagonal is L's diagonal)
    # and U's off-diagonals in the upper part. U's diagonal is given by dU.
    
def forward_substitution(A, b):
    """
    Solve L y = b, where L is lower triangular stored in A (including its diagonal).
    
    Returns:
       y: solution vector.
    """
    n = A.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += A[i, j] * y[j]
        y[i] = (b[i] - s) / A[i, i]
    return y

def backward_substitution(A, dU, y):
    """
    Solve U x = y, where U is upper triangular.
    U's off-diagonals are stored in A (for i < j) and the diagonal entries are in dU.
    
    Returns:
       x: solution vector.
    """
    n = A.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        s = 0.0
        for j in range(i+1, n):
            s += A[i, j] * x[j]
        if abs(dU[i]) < 1e-15:
            raise ValueError(f"Zero pivot encountered in U at index {i}")
        x[i] = (y[i] - s) / dU[i]
    return x

def compute_determinant(A, dU):
    """
    Given the LU decomposition stored in A (L in lower part) and dU (diagonal of U),
    compute det(A) = (prod_i L[i,i])*(prod_i dU[i])
    """
    n = A.shape[0]
    detL = 1.0
    for i in range(n):
        detL *= A[i, i]
    detU = np.prod(dU)
    return detL * detU

def run_inplace_version(A_init, b, dU, eps):
    """
    Perform the in-place LU decomposition, solve the system, verify the residual,
    and compare with library solutions.
    """
    n = A_init.shape[0]
    A = A_init.copy()  # will be modified
    lu_decomposition_inplace(A, dU, eps)
    # Compute determinant
    detA = compute_determinant(A, dU)
    print("Determinant of A =", detA)
    
    # Solve using LU factors:
    y = forward_substitution(A, b)
    x_LU = backward_substitution(A, dU, y)
    # Verify residual norm using the original A_init:
    res_norm = np.linalg.norm(A_init @ x_LU - b, 2)
    print("Residual norm ||A_init*x_LU - b|| =", res_norm)
    
    # Solve using NumPy's solver and compute inverse:
    x_lib = np.linalg.solve(A_init, b)
    A_inv_lib = np.linalg.inv(A_init)
    
    diff1 = np.linalg.norm(x_LU - x_lib, 2)
    diff2 = np.linalg.norm(x_LU - A_inv_lib @ b, 2)
    print("||x_LU - x_lib|| =", diff1)
    print("||x_LU - A_inv_lib*b|| =", diff2)
    
    return x_LU

# =================== BONUS VERSION ======================

def idx_lower(i, j):
    """Mapping for L: 0 <= j <= i < n. For a given n, index = i*(i+1)//2 + j."""
    return i*(i+1)//2 + j

def idx_upper(i, j, n):
    """Mapping for U: 0 <= i <= j < n.
       The number of elements in row i of the upper triangular part is (n-i).
       Index = sum_{k=0}^{i-1} (n-k) + (j-i).
    """
    return (i * n - (i*(i-1))//2) + (j - i)

def lu_decomposition_bonus(A, dU, eps):
    """
    LU decomposition using memory restrictions:
      - A remains unchanged.
      - L and U are stored in two 1-D vectors of size n(n+1)/2.
    The algorithm is analogous to the in-place one.
    
    Returns:
        L_vec: vector storing L (for indices with i>=j)
        U_vec: vector storing U (for indices with i<=j), with U[i,i]=dU[i]
    """
    n = A.shape[0]
    size = n*(n+1)//2
    L_vec = np.zeros(size)
    U_vec = np.zeros(size)
    
    # Process row by row:
    for p in range(n):
        # Compute L[p, j] for j=0,...,p-1:
        for j in range(p):
            sum_LU = 0.0
            for k in range(j):
                sum_LU += L_vec[idx_lower(p, k)] * U_vec[idx_upper(k, j, n)]
            # U[j,j] = dU[j]
            if abs(dU[j]) < eps:
                raise ValueError(f"Zero pivot encountered in U at index {j}")
            L_val = (A[p, j] - sum_LU) / dU[j]
            L_vec[idx_lower(p, j)] = L_val
        # Diagonal L[p,p]:
        sum_LU = 0.0
        for k in range(p):
            sum_LU += L_vec[idx_lower(p, k)] * U_vec[idx_upper(k, p, n)]
        L_diag = A[p, p] - sum_LU
        if abs(L_diag) < eps:
            raise ValueError(f"Zero pivot encountered in L at index {p}")
        L_vec[idx_lower(p, p)] = L_diag
        # Compute U[p, j] for j = p+1,..., n-1:
        for j in range(p+1, n):
            sum_LU = 0.0
            for k in range(p):
                sum_LU += L_vec[idx_lower(p, k)] * U_vec[idx_upper(k, j, n)]
            U_val = (A[p, j] - sum_LU) / L_diag
            U_vec[idx_upper(p, j, n)] = U_val
        # Set diagonal of U:
        U_vec[idx_upper(p, p, n)] = dU[p]
    return L_vec, U_vec

def forward_substitution_bonus(L_vec, b, n):
    """Solve L y = b using L_vec."""
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L_vec[idx_lower(i, j)] * y[j]
        y[i] = (b[i] - s) / L_vec[idx_lower(i, i)]
    return y

def backward_substitution_bonus(U_vec, dU, y, n):
    """Solve U x = y using U_vec. (U[i,i] is stored in dU or in U_vec via idx_upper(i,i))."""
    x = np.zeros(n)
    for i in reversed(range(n)):
        s = 0.0
        for j in range(i+1, n):
            s += U_vec[idx_upper(i, j, n)] * x[j]
        # U[i,i] = dU[i] (or U_vec[idx_upper(i,i,n)])
        if abs(dU[i]) < 1e-15:
            raise ValueError(f"Zero pivot encountered in U at index {i}")
        x[i] = (y[i] - s) / dU[i]
    return x

def reconstruct_LU(L_vec, U_vec, dU, n):
    """
    Reconstruct full matrix product L*U from the stored vectors.
    Returns an (n,n) matrix.
    """
    LU = np.zeros((n,n))
    # Build L and U in full matrix form from vector storage:
    L_full = np.zeros((n,n))
    U_full = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            L_full[i,j] = L_vec[idx_lower(i, j)]
        for j in range(i, n):
            if i==j:
                U_full[i,j] = dU[i]
            else:
                U_full[i,j] = U_vec[idx_upper(i, j, n)]
    LU = L_full @ U_full
    return LU

def run_bonus_version(A, b, dU, eps):
    """
    Run the bonus version: using one matrix A (unchanged) and two vectors for L and U.
    Solve the system and display the LU product.
    """
    n = A.shape[0]
    L_vec, U_vec = lu_decomposition_bonus(A, dU, eps)
    y = forward_substitution_bonus(L_vec, b, n)
    x_LU_bonus = backward_substitution_bonus(U_vec, dU, y, n)
    LU_prod = reconstruct_LU(L_vec, U_vec, dU, n)
    print("Solution x (bonus version):", x_LU_bonus)
    print("Reconstructed LU product (should approximate A):")
    print(LU_prod)
    # Optionally, display norm difference between A and LU_prod:
    diff_norm = np.linalg.norm(A - LU_prod, 2)
    print("||A - L*U|| =", diff_norm)
    return x_LU_bonus

# =================== Main Testing ======================

if __name__ == "__main__":
    # Example input:
    # (You can replace these with inputs read from file or keyboard)
    n = 3
    eps = 1e-8
    # Define A, b, and dU as in the first example from the document:
    A_init = np.array([[4.0, 2.0, 3.0],
                       [2.0, 7.0, 5.5],
                       [6.0, 3.0, 12.5]])
    b = np.array([21.6, 33.6, 51.6])
    dU = np.array([2.0, 3.0, 4.0])
    
    print("------ In-Place LU Decomposition Version ------")
    x_LU = run_inplace_version(A_init, b, dU, eps)
    print("Solution x (in-place version):", x_LU)
    
    print("\n------ Bonus Version (Memory-Restricted) ------")
    # For bonus, use the same A_init (unchanged) and same dU, b
    x_bonus = run_bonus_version(A_init, b, dU, eps)
