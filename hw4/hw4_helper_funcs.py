import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import svd
from scipy.optimize import minimize

import cvxpy as cp

import matplotlib.pyplot as plt





def first_deriv_mat(n):

    # Initialize the n x n matrix with zeros
    L = np.zeros((n, n))
    
    # Fill in the main part of the matrix
    for i in range(n - 1):
        L[i, i] = -1
        L[i, i + 1] = 1
    
    # Set the top-right and bottom-left elements
    L[0, n - 1] = 1
    L[n - 1, 0] = -1
    
    return L



def convmtx(h, n):
    """
    Create a convolution matrix (Toeplitz matrix) from vector h.
    
    Parameters:
    h (array-like): Input vector (the filter).
    n (int): The number of rows for the resulting convolution matrix.
    
    Returns:
    numpy.ndarray: The convolution matrix.
    """
    h = np.asarray(h)
    # Create the first column: h followed by zeros
    col = np.concatenate([h, np.zeros(n - 1)])
    
    # Create the first row: h[0] followed by zeros
    row = np.concatenate([[h[0]], np.zeros(n - 1)])
    
    # Generate the Toeplitz matrix
    conv_matrix = toeplitz(col, row)
    
    return conv_matrix



def generate_sparse_vector(n, s):
    """
    Generates a sparse vector of dimension n with s non-zero entries.

    Parameters:
        n (int): The dimension of the vector.
        s (int): The number of non-zero entries (sparsity).

    Returns:
        numpy.ndarray: A vector of length n with s non-zero components.
    """
    np.random.seed(0)

    # Randomly shuffle indices and select the first `s` indices
    idx = np.random.permutation(n)[:s]
    
    # Initialize `f` as a zero vector and set the selected indices to random values
    f = np.zeros(n)
    f[idx] = np.random.randn(s)
    
    return f




def differencing_matrix(n):
    """
    Generates an n x n tridiagonal matrix with -1 on the diagonal, 
    1 on the superdiagonal and subdiagonal.
    
    Parameters:
        n (int): The dimension of the matrix.
        
    Returns:
        numpy.ndarray: The generated n x n matrix.
    """
    # Create a matrix of zeros
    L = np.zeros((n, n))
    
    # Set the diagonal entries to -1
    np.fill_diagonal(L, -1)
    

    # Set the subdiagonal entries to 1
    np.fill_diagonal(L[:, 1:], 1)

    L[-1,0] = 1.0
    
    return L



def least_squares_solver(A, b):
    """Solves the least squares problem argmin || A x - b ||_2.
    """
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    return x


def solve_tikhonov(A, b, L, lambdah):
    """
    Solves the regularized least squares problem:
        argmin_x || A x - b ||_2^2 + lambdah^2 || L x ||_2^2
    
    Parameters:
        A (numpy.ndarray): The matrix A.
        b (numpy.ndarray): The vector b.
        L (numpy.ndarray): The regularization matrix L.
        lambdah (float): The regularization parameter lambda.

    Returns:
        x (numpy.ndarray): The solution vector x.
    """
    # Form the regularized normal equations
    A_aug = np.vstack((A, lambdah * L))
    b_aug = np.concatenate((b, np.zeros(L.shape[0])))

    # Solve the least squares problem with the augmented system
    x, residuals, rank, s = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    
    return x




def solve_l1(A, b, L, lambdah):
    """Solves the problem argmin_x || A x - b ||_2^2 + lambda^2 || L x ||_1
    """

    _x = cp.Variable(A.shape[1])
    L2_term = cp.sum_squares(A @ _x - b)
    L1_term = cp.norm1(L @ _x)
    objective = cp.Minimize(L2_term + ( (lambdah**2)*L1_term ) )
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, max_iters=10000, eps=1e-10)
    optimal_x = _x.value

    return optimal_x




def shaw(n):
    """
    Test problem: one-dimensional image restoration model.

    Discretization of a first-kind Fredholm integral equation with
    [-pi/2, pi/2] as both integration intervals.

    Parameters:
    n -- Order of the matrix (must be even).

    Returns:
    A -- Discretized matrix from the Fredholm integral equation.
    b -- Right-hand side vector.
    x -- Solution vector.

    Reference: C. B. Shaw, Jr., "Improvements of the resolution of
    an instrument by numerical solution of an integral equation",
    J. Math. Anal. Appl. 37 (1972), 83-112.

    Adapted from shaw.m in regtools, by Per Christian Hansen.
    """
    
    assert n%2 == 0, "input n must be even!"
    nhalf = int(n/2.0)

    # Initialization.
    h = np.pi/n
    A = np.zeros((n,n))

    # Compute the matrix A.
    co = np.cos(-(np.pi/2) + (np.arange(0.5, n-0.5+1, 1)*h)  )
    psi = np.pi*np.sin(-(np.pi/2) + (np.arange(0.5, n-0.5+1, 1)*h))

    for i in range(nhalf):
        for j in range(i,(n-1)-i):
            ss = psi[i] + psi[j]
            A[i,j] = ((co[i] + co[j])*np.sin(ss)/ss)**2
            A[(n-1)-j,(n-1)-i] = A[i,j]
         
        A[i, (n-1)-i] = (2*co[i])**2

    A = A.copy() + np.triu(A,1).T
    A = A*h

    # Compute the vectors x and b
    a1 = 2 
    c1 = 6
    t1 =  .8
    a2 = 1
    c2 = 2
    t2 = -.5

    x = ( a1*np.exp(-c1*(-np.pi/2 + (np.arange(0.5, n-0.5+1, 1)*h) - t1)**2) 
        + a2*np.exp(-c2*(-np.pi/2 + (np.arange(0.5, n-0.5+1, 1)*h) - t2)**2) )
                      
    b = A @ x

    return A, b, x







def csvd(A, tst=None):
    """
    Computes the compact form of the Singular Value Decomposition (SVD) of matrix A:
        A = U @ np.diag(s) @ V.T
    where U, s, V are the SVD components with the following dimensions:
        U  is m-by-min(m, n)
        s  is min(m, n)-by-1
        V  is n-by-min(m, n).
    
    If a second argument is provided, the full U and V matrices are returned.
    
    Parameters:
    A : array_like
        Input matrix of shape (m, n).
    tst : optional
        If provided, returns the full SVD matrices.
    
    Returns:
    U : ndarray
        Left singular vectors.
    s : ndarray
        Singular values.
    V : ndarray
        Right singular vectors.
    """
    m, n = A.shape
    if tst is None:
        # Compact SVD
        if m >= n:
            U, s, Vh = svd(A, full_matrices=False)
        else:
            Vh, s, U = svd(A.T, full_matrices=False)
            Vh = Vh.T
    else:
        # Full SVD
        U, s, Vh = svd(A, full_matrices=True)

    return U, s, Vh.T




def determine_optimal_lambdah(A, b, x_true):
    """Attempts to find the optimal tikhonov parameter \lambda in 
    || A x - b ||_2^2 + \lambda^2 || x ||_2^2.

    Returns x_lambda, lambda (not lambda^2).
    """
    
    tikh_sol = lambda log_lambdah_sq: np.linalg.solve(A.T @ A + np.exp(log_lambdah_sq)*np.eye(A.shape[1]), A.T @ b)
    obj_func = lambda log_lambdah_sq: np.linalg.norm(tikh_sol(log_lambdah_sq) - x_true)
    result = minimize(obj_func, 0.0, tol=1e-10)

    return tikh_sol(result.x), np.sqrt(np.exp(result.x))[0]



def discrep(U, s, V, b, delta, x_0=None):
    m, n = U.shape[0], V.shape[1]
    p, ps = s.shape if s.ndim > 1 else (s.size, 1)
    delta = np.atleast_1d(delta)  # Ensure delta is a 1D array
    ld = delta.size
    x_delta = np.zeros((n, ld))
    lambdah = np.zeros(ld)
    rho = np.zeros(p)

    if np.any(delta < 0):
        raise ValueError("Illegal inequality constraint delta")

    if x_0 is None:
        x_0 = np.zeros(n)

    omega = V.T @ x_0 if ps == 1 else np.linalg.solve(V, x_0)

    # Compute residual norms corresponding to TSVD/TGSVD
    beta = U.T @ b
    delta_0 = np.linalg.norm(b - U @ beta)
    
    if ps == 1:
        rho[p - 1] = delta_0 ** 2
        for i in range(p - 2, -1, -1):
            rho[i] = rho[i + 1] + (beta[i + 1] - s[i + 1] * omega[i + 1]) ** 2
    else:
        rho[0] = delta_0 ** 2
        for i in range(p - 1):
            rho[i + 1] = rho[i] + (beta[i] - s[i, 0] * omega[i]) ** 2

    if np.any(delta < delta_0):
        raise ValueError("Irrelevant delta < || (I - U * U.T) * b ||")

    if ps == 1:
        s2 = s ** 2
        for k in range(ld):
            if delta[k] ** 2 >= np.linalg.norm(beta - s * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = x_0
            else:
                kmin = np.argmin(np.abs(rho - delta[k] ** 2))
                lambda_0 = s[kmin]
                lambdah[k] = newton(lambda_0, delta[k], s, beta, omega, delta_0)
                e = s / (s2 + lambdah[k] ** 2)
                f = s * e
                x_delta[:, k] = V[:, :p] @ (e * beta + (1 - f) * omega)

    elif m >= n:
        omega = omega[:p]
        gamma = s[:, 0] / s[:, 1]
        x_u = V[:, p:] @ beta[p:]
        for k in range(ld):
            if delta[k] ** 2 >= np.linalg.norm(beta[:p] - s[:, 0] * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = V @ np.hstack([omega, U[:, p:].T @ b])
            else:
                kmin = np.argmin(np.abs(rho - delta[k] ** 2))
                lambda_0 = gamma[kmin]
                lambdah[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma / (gamma ** 2 + lambdah[k] ** 2)
                f = gamma * e
                x_delta[:, k] = V[:, :p] @ (e * beta[:p] / s[:, 1] + (1 - f) * s[:, 1] * omega) + x_u

    else:
        omega = omega[:p]
        gamma = s[:, 0] / s[:, 1]
        x_u = V[:, p:m] @ beta[p:m]
        for k in range(ld):
            if delta[k] ** 2 >= np.linalg.norm(beta[:p] - s[:, 0] * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = V @ np.hstack([omega, U[:, p:m].T @ b])
            else:
                kmin = np.argmin(np.abs(rho - delta[k] ** 2))
                lambda_0 = gamma[kmin]
                lambdah[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma / (gamma ** 2 + lambdah[k] ** 2)
                f = gamma * e
                x_delta[:, k] = V[:, :p] @ (e * beta[:p] / s[:, 1] + (1 - f) * s[:, 1] * omega) + x_u

    return x_delta, lambdah[0]


def newton(lambda_0, delta, s, beta, omega, delta_0):
    thr = np.sqrt(np.finfo(float).eps)
    it_max = 50

    if lambda_0 < 0:
        raise ValueError("Initial guess lambda_0 must be nonnegative")

    p, ps = s.shape if s.ndim > 1 else (s.size, 1)
    if ps == 2:
        sigma = s[:, 0]
        s = s[:, 0] / s[:, 1]
    s2 = s ** 2

    lambda_val = lambda_0
    for it in range(it_max):
        f = s2 / (s2 + lambda_val ** 2)
        r = (1 - f) * (beta - s * omega) if ps == 1 else (1 - f) * (beta - sigma * omega)
        z = f * r
        step = (lambda_val / 4) * (r @ r + (delta_0 + delta) * (delta_0 - delta)) / (z @ r)
        
        if abs(step) <= thr * lambda_val or abs(step) <= thr:
            return lambda_val
        
        lambda_val -= step
        if lambda_val < 0:
            lambda_val = 0.5 * lambda_0
            lambda_0 *= 0.5

    raise RuntimeError(f"Max number of iterations ({it_max}) reached")






def piecewise_constant_function(n):
    """
    Piecewise function f(t) defined as:
    - 1 if -1/4 < t <= 0
    - 2 if 1/2 <= t <= 7/8
    - 0 otherwise
    
    Parameters:
        t (float): The input value.
        
    Returns:
        int: The output of the function based on the value of t.
    """

    grid = np.asarray([-1.0 + 2*j/n for j in range(n)])

    def _f(t):
        if -1/4 < t <= 0:
            return 1
        elif 1/2 <= t <= 7/8:
            return 2
        else:
            return 0
        
    result = np.zeros_like(grid)
    for j, item in enumerate(grid):
        result[j] = _f(item)

    return result







def sawtooth_function(n):
    """
    Piecewise function f(t) defined as:
    - 1 if -1/4 < t <= 0
    - 2 if 1/2 <= t <= 7/8
    - 0 otherwise
    
    Parameters:
        t (float): The input value.
        
    Returns:
        int: The output of the function based on the value of t.
    """

    grid = np.asarray([-1.0 + 2*j/n for j in range(n)])

    def _f(t):
        if -1/4 < t <= 0:
            return 1
        elif 1/2 <= t <= 7/8:
            return 2
        else:
            return 0
        
    result = np.zeros_like(grid)
    for j, item in enumerate(grid):
        result[j] = _f(item)

    return result




def sawtooth_function(n):
    """
    Piecewise function f(t) defined as:
    - -pi - t if -pi < t <= 0
    - pi - t if t > 0
    
    Parameters:
        t (float): The input value.
        
    Returns:
        float: The output of the function based on the value of t.
    """

    grid = np.asarray([-1.0 + 2*j/n for j in range(n)])

    def _f(t):
        if -np.pi < t <= 0:
            return -np.pi - t
        elif t > 0:
            return np.pi - t
        else:
            # Return None or raise an error if t is not in the specified domain
            return None
        
    result = np.zeros_like(grid)
    for j, item in enumerate(grid):
        result[j] = _f(item)

    return result


