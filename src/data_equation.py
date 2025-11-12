import sys
import numpy as np
import scipy.linalg as LA


def O(A, C, n):
    '''
    Create the extended observability matrix used in the data equation.

    Parameters:

    A : numpy.ndarray
        The state matrix of the state-space system
    C : numpy.ndarray
        The observation matrix of the state-space system
    n : float
        number of measurements

    Returns:

    O : numpy.ndarray, shape(n, number of state variables)
        The extended observability matrix
    '''
    A_power = np.copy(A)
    O = np.vstack((np.copy(C), C @ A))

    for k in range(n-2):
        A_power = A_power @ A
        O = np.vstack((O, C @ A_power))

    return O


def gamma(A, B, C, n):
    '''
    Create the impulse response matrix used in the data equation.

    Parameters:

    A : numpy.ndarray
        The state matrix of the state-space system
    B : numpy.ndarray
        The input matrix of the state-space system
    C : numpy.ndarray
        The observation matrix of the state-space system
    n : float
        number of measurements

    Returns:

    gamma : numpy.ndarray, shape(n*number of state variables, n*number of state variables)
        The impulse response matrix
    '''
    A_power = np.copy(A)
    Z = np.zeros((C @ B).shape)

    # first column
    gamma_column_first = np.vstack((
        Z,
        C @ B,
        C @ A @ B
    ))
    for _ in range(n-3):
        A_power = A_power @ A
        gamma_column_first = np.vstack((gamma_column_first, C @ A_power @ B))

    # build complete matrix, column by column, from left to right
    gamma = np.copy(gamma_column_first)
    current_column = 1
    for _ in range(1, n):
        gamma_rows = Z

        # first add zero matrices
        for _ in range(current_column):
            gamma_rows = np.vstack((gamma_rows, Z))

        # then add the impulse responses
        A_power2 = np.copy(A)

        if current_column < (n-2):
            gamma_rows = np.vstack((
                gamma_rows,
                C @ B,
                C @ A @ B # these must not be added to the last and the second to last columns
            ))

        if current_column == (n-2):
            gamma_rows = np.vstack((
                gamma_rows,
                C @ B # this has to be added to the end of the second to last column
            ))

        for _ in range(n-current_column-3):
            A_power2 = A_power2 @ A
            gamma_rows = np.vstack((gamma_rows, C @ A_power2 @ B))

        # add column on the right hand side
        gamma = np.hstack((gamma, gamma_rows))
        current_column += 1

    return gamma


def second_difference_matrix(n, m):
    # Second difference matrix for two inputs
    D2 = np.eye(n*m) - 2*np.eye(n*m, k=2) + np.eye(n*m, k=4)

    # delete incomplete rows
    D2 = D2[:-2*m, :]

    return D2


def dft_matrix(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)

    return W / np.sqrt(N)  # / N


def get_data_equation_matrices(A, B, C, D, n, bs):
    D2 = second_difference_matrix(bs, B.shape[1])
    O_mat = O(A, C, bs)
    G = gamma(A, B, C, bs)
    L = np.eye(bs*B.shape[1])

    return O_mat, G, D2, L


def ell2_analytical(ss, ss2, measurements, batch_size, overlap, times, lam1=0, lam2=0.1, use_trend_filter=False, print_bar=True, full_scale=False):
    """
    Analytical solution of the l2 regularized LS problem.
    Minimizes the sum of squared residuals, including an l2 constraint.
    """
    dt = np.mean(np.diff(times))
    n = len(times)
    bs = batch_size + 2*overlap
    loop_len = int(n/batch_size)

    A, B, C, D, dt_ss = ss  # state space model
    O_mat, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)  # data equation matrices

    if use_trend_filter:
        regul_matrix = D2 # regularization matrix
    else:
        regul_matrix = L

    H = np.hstack([O_mat, G])  # extended observation and impulse response matrix

    # initial state regularization, Z: no regularization, I: yes regularization
    Z = np.zeros((regul_matrix.shape[0], O_mat.shape[1]))
    S = (np.eye(O_mat.shape[1])-np.eye(O_mat.shape[1], k=1))
    # initial state regularization is set to zero at gear locations and
    # where state quantity changes (tau -> theta_dot)
    if full_scale:
        S[11] *= 0  # gear 1 torque
        S[14] *= 0  # gear 2 torque
        S[16] *= 0  # tau_n - theta_dot_1
        S[32] *= 0  # gear 1 speed
        S[34] *= 0  # gear 2 speed
    else:
        S[10] *= 0  # gear 1 torque
        S[15] *= 0  # gear 2 torque
        S[20] *= 0  # tau_n - theta_dot_1
        S[32] *= 0  # gear 1 speed
        S[37] *= 0  # gear 2 speed

    Id = np.vstack([
        S,
        np.zeros((regul_matrix.shape[0]-O_mat.shape[1], O_mat.shape[1]))
    ])
    M = np.vstack([
        lam1*(np.hstack([Id, np.zeros_like(regul_matrix)])),
        lam2*(np.hstack([Z, regul_matrix]))
    ])  # extended regularization matrix

    # measurement noise covariance matrix
    if full_scale:
        R = np.eye(2)
        n_states = 35
    else:
        R = np.diag([0.10, 0.10, 0.10])  # NOTE: for simulation
        # R = np.diag([0.05, 0.10, 0.20])
        n_states = 43
    R_inv = LA.inv(R)
    I = np.eye(bs)
    # measurement noise covariance assembled as a diagonal block matrix
    WR = np.kron(I, R_inv)

    LS = LA.inv(H.T @ WR @ H + (M.T @ M)) @ H.T @ WR

    input_estimates = []
    state_estimates = []

    A2, B2, c, d, dt_ss2 = ss2 # for state reconstruction
    O_mat2, G2, D22, L2 = get_data_equation_matrices(A2, B2, c, d, n, bs)  # data equation matrices

    for i in range(loop_len):
        if i == 0:
            batch = measurements[:bs,:]
        elif i == loop_len-1 and full_scale:
            batch = np.zeros((bs, measurements.shape[1]))
            # zero padding to finish estimation loop correctly
        else:
            batch = measurements[i*batch_size-overlap:(i+1)*batch_size+overlap,:]

        y = batch.reshape(-1,1)

        estimate = LS @ y
        input_estimates.append(estimate)

        state_estimate = O_mat2 @ estimate[:n_states].reshape(-1,1) + G2 @ estimate[n_states:].reshape(-1,1)
        state_estimates.append(state_estimate)

    return input_estimates, state_estimates


def progressbar(it, prefix="", size=1, out=sys.stdout, show_print=False):
    """
    A function used to display a progress bar in the console.
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    if show_print:
        print("\n", flush=True, file=out)
