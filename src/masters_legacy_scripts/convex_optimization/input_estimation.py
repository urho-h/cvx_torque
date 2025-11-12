import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlsim
import pickle

import cvxpy as cp

from . import data_equation as de


def tikhonov_problem(meas, obsrv, gamm, regu, initial_state=None, lam=1, cmplx=False):
    '''
    This function uses the cvxpy library to solve a Tikhonov regularization problem.
    '''
    d = cp.Variable((gamm.shape[1], 1), complex=cmplx)

    if initial_state is None:
        x = cp.Variable((obsrv.shape[1], 1), complex=cmplx)
    else:
        x = initial_state

    measurements = cp.Parameter(meas.shape)
    measurements.value = meas

    objective = cp.Minimize(cp.sum_squares(measurements - obsrv @ x - gamm @ d) + lam * cp.sum_squares(regu @ d))

    prob = cp.Problem(objective)
    prob.solve()

    if initial_state is None:
        x_value = x.value
    else:
        x_value = initial_state

    return d.value, x_value


def lasso_problem(meas, obsrv, gamm, regu, initial_state=None, lam=1, cmplx=False):
    '''
    This function uses the cvxpy library to solve a LASSO problem.
    '''
    d = cp.Variable((gamm.shape[1], 1), complex=cmplx)

    if initial_state is None:
        x = cp.Variable((obsrv.shape[1], 1), complex=cmplx)
    else:
        x = initial_state

    measurements = cp.Parameter(meas.shape)
    measurements.value = meas

    objective = cp.Minimize(cp.sum_squares(measurements - obsrv @ x - gamm @ d) + lam * cp.pnorm(regu @ d, 1))

    prob = cp.Problem(objective)
    prob.solve()

    if initial_state is None:
        x_value = x.value
    else:
        x_value = initial_state

    return d.value, x_value


def elastic_net_problem(meas, obsrv, gamm, regu, initial_state=None, lam1=1, lam2=1, cmplx=False):
    '''
    This function uses the cvxpy library to solve an elastic net problem.
    '''
    d = cp.Variable((gamm.shape[1], 1), complex=cmplx)

    if initial_state is None:
        x = cp.Variable((obsrv.shape[1], 1), complex=cmplx)
    else:
        x = initial_state

    measurements = cp.Parameter(meas.shape)
    measurements.value = meas

    objective = cp.Minimize(
        cp.sum_squares(measurements - obsrv @ x - gamm @ d) + lam1 * cp.sum_squares(regu @ d) + lam2 * cp.pnorm(regu @ d, 1)
    )

    prob = cp.Problem(objective)
    prob.solve()

    if initial_state is None:
        x_value = x.value
    else:
        x_value = initial_state

    return d.value, x_value


def get_data_equation_matrices(A, B, C, D, n, bs):
    D2 = de.second_difference_matrix(bs, B.shape[1])
    O = de.O(A, C, bs)
    G = de.gamma(A, B, C, bs)
    L = np.eye(bs*B.shape[1])
    W = de.dft_matrix(4)

    return O, G, D2, L, W


def progressbar(it, prefix="", size=60, out=sys.stdout, show_print=False):
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


def L_curve(sys, measurements, times, lambdas, use_zero_init=True, use_l1=False, use_trend=False):
    dt = np.mean(np.diff(times))
    bs = len(times)
    n = len(times)

    A, B, C, D = sys
    O, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_trend:
        regularization = D2
    else:
        regularization = L

    if use_zero_init:
        x_init = np.zeros((O.shape[1], 1))
    else:
        x_init = None

    input_estimates = []

    y = measurements.reshape(-1,1)

    for i in progressbar(range(len(lambdas)), "Calculating estimates :", len(lambdas)):
        if use_l1:
            estimate, x_init = tikhonov_problem(y, O, G, regularization, initial_state=x_init, lam=lambdas[i])
        else:
            estimate, x_init = lasso_problem(y, O, G, regularization, initial_state=x_init, lam=lambdas[i])
        input_estimates.append(estimate)

    norm, res_norm = [], []
    for i in range(len(lambdas)):
        res_norm.append(np.linalg.norm(y - G @ input_estimates[i]))
        if use_l1:
            norm.append(np.linalg.norm(regularization @ input_estimates[i], ord=1))
        else:
            norm.append(np.linalg.norm(regularization @ input_estimates[i]))

    return norm, res_norm


def pareto_curve(sys, load, measurements, times, lambdas, use_zero_init=True, use_trend=False):
    dt = np.mean(np.diff(times))
    bs = len(times)
    n = len(times)

    A, B, C, D = sys
    O, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_trend:
        regularization = D2
    else:
        regularization = L

    if use_zero_init:
        x_init = np.zeros((O.shape[1], 1))
    else:
        x_init = None

    y = measurements.reshape(-1,1)
    u = load.reshape(-1,1)

    input_estimates = []

    for i in progressbar(range(len(lambdas)), "Calculating norms :", len(lambdas)):
        intermediate = []
        for j in range(len(lambdas)):
            estimate, _ = elastic_net_problem(
                y,
                O,
                G,
                regularization,
                initial_state=x_init,
                lam1=lambdas[i],
                lam2=lambdas[j]
            )
            intermediate.append(estimate)
        input_estimates.append(intermediate)

    norms_l1, norms_l2, res_norms = [], [], []

    for i in progressbar(range(len(lambdas)), "Calculating norms :", len(lambdas)):
        norm_l1, norm_l2, res_norm = [], [], []
        for j in range(len(lambdas)):
            norm_l1.append(np.linalg.norm(y - G @ input_estimates[i][j], ord=1))
            norm_l2.append(np.linalg.norm(y - G @ input_estimates[i][j]))
            res_norm.append(np.linalg.norm(regularization @ input_estimates[i][j]))
        norms_l1.append(norm_l1)
        norms_l2.append(norm_l2)
        res_norms.append(res_norm)

    return norms_l1, norms_l2, res_norms


def data_eq_simulation(sys, times, load, bs):
    dt = np.mean(np.diff(times))
    n = len(times)

    A, B, C, D = sys
    omat = de.O(A, C, bs)
    gmat = de.gamma(A, B, C, bs)

    u = load.T.reshape(-1,1)

    return gmat @ u


def estimate_input(sys, measurements, batch_size, overlap, times, lam=0.1, lam2=0.1, use_zero_init=True, use_lasso=False, use_elastic_net=False, use_trend_filter=False, use_dft=False, pickle_data=False, fn="input_estimates_"):
    dt = np.mean(np.diff(times))
    n = len(times)
    bs = batch_size + 2*overlap
    loop_len = int(n/batch_size)

    A, B, C, D = sys
    O, G, D2, L, W = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_trend_filter:
        regul_matrix = D2
    elif use_dft:
        regul_matrix = W
    else:
        regul_matrix = L

    if use_zero_init:
        x_init = np.zeros((O.shape[1], 1))
    else:
        x_init = None

    input_estimates = []

    # for initial state estimation
    C_full = np.eye(B.shape[0])
    omat = de.O(A, C_full, bs)
    gmat = de.gamma(A, B, C_full, bs)

    for i in progressbar(range(loop_len), "Calculating estimates: ", loop_len):
        if i == 0:
            batch = measurements[:bs,:]
        elif i == loop_len-1:
            batch = np.zeros((bs, measurements.shape[1]))
            # zero padding to finish estimation loop correctly
        else:
            batch = measurements[i*batch_size-overlap:(i+1)*batch_size+overlap,:]

        y = batch.reshape(-1,1)

        if use_lasso:
            estimate, x_init = lasso_problem(y, O, G, regul_matrix, initial_state=x_init, lam=lam)
        elif use_elastic_net:
            estimate, x_init = elastic_net_problem(y, O, G, regul_matrix, initial_state=x_init, lam1=lam, lam2=lam2)
        else:
            estimate, x_init = tikhonov_problem(y, O, G, regul_matrix, initial_state=x_init, lam=lam)

        x_est = omat @ x_init + gmat @ estimate

        x_init = x_est[-A.shape[0]:,:]

        input_estimates.append(estimate)

        if pickle_data:
            with open(fn + str(i) + ".pickle", 'wb') as handle:
                pickle.dump([estimate, x_est], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return input_estimates


def ell2_analytical(ss, measurements, batch_size, overlap, times, lam=0.1, use_trend_filter=False, print_bar=True, pickle_data=False, fn='input_estimates'):
    """
    Analytical solution of the l2 regularized LS problem.
    Minimizes the sum of squared residuals, including an l2 constraint.
    """
    dt = np.mean(np.diff(times))
    n = len(times)
    bs = batch_size + 2*overlap
    loop_len = int(n/batch_size)

    A, B, C, D = ss  # state space model
    O_mat, G, D2, L, W = get_data_equation_matrices(A, B, C, D, n, bs)  # data equation matrices

    if use_trend_filter:
        regul_matrix = D2 # regularization matrix
    else:
        regul_matrix = L

    H = np.hstack([O_mat, G])  # extended observation and impulse response matrix
    M = np.hstack([np.zeros((regul_matrix.shape[0], O_mat.shape[1])), regul_matrix])  # extended regularization matrix
    Ht = H.T
    HtH = Ht @ H
    MtM = M.T @ M
    # Least-squares solution: u_hat = LS @ y
    LS = LA.inv(HtH + lam*MtM) @ Ht

    input_estimates = []

    for i in range(loop_len):
        if i == 0:
            batch = measurements[:bs,:]
        elif i == loop_len-1:
            batch = np.zeros((bs, measurements.shape[1]))
            # zero padding to finish estimation loop correctly
        else:
            batch = measurements[i*batch_size-overlap:(i+1)*batch_size+overlap,:]

        y = batch.reshape(-1,1)
        estimate = LS @ y
        input_estimates.append(estimate)  # estimate includes initial state estimate

        if pickle_data:
            with open(fn + str(i) + ".pickle", 'wb') as handle:
                pickle.dump([estimate], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return input_estimates


def ell2_analytical_with_covariance(ss, measurements, batch_size, overlap, times, lam=0.1, use_trend_filter=False, print_bar=True):
    """
    Analytical solution of the l2 regularized LS problem.
    Minimizes the sum of squared residuals, including an l2 constraint and known covariance for input and output.
    Initial state is always assumed zero.
    """
    dt = np.mean(np.diff(times))
    n = len(times)
    bs = batch_size + 2*overlap
    loop_len = int(n/batch_size)

    A, B, C, D = ss  # state space model
    O_mat, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)  # data equation matrices

    if use_trend_filter:
        regul_matrix = D2 # regularization matrix
    else:
        regul_matrix = L

    # measurement noise covariance matrix
    R = np.diag([0.05, 0.10, 0.20])
    #R = np.diag([0.03, 0.20, 0.01])
    R_inv = LA.inv(R)
    I = np.eye(bs)
    # measurement noise covariance assembled as a diagonal block matrix
    WR = np.kron(I, R_inv)
    # Least-squares solution: u_hat = LS @ y
    LS = LA.inv(G.T @ WR @ G + lam*(regul_matrix.T@regul_matrix)) @ G.T @ WR

    input_estimates = []

    for i in progressbar(range(loop_len), "Calculating estimates: ", loop_len, show_print=print_bar):
        if i == 0:
            batch = measurements[:bs,:]
        elif i == loop_len-1:
            batch = np.zeros((bs, measurements.shape[1]))
            # zero padding to finish estimation loop correctly
        else:
            batch = measurements[i*batch_size-overlap:(i+1)*batch_size+overlap,:]

        y = batch.reshape(-1,1)

        estimate = LS @ y

        input_estimates.append(estimate)

    return input_estimates


def conjugate_gradient(ss, measurements, batch_size, overlap, times, tol=0.1, max_iters=1000, print_bar=True, pickle_data=False, fn='input_estimates'):
    """
    Conjugate gradient method.
    """
    dt = np.mean(np.diff(times))
    n = len(times)
    bs = batch_size + 2*overlap
    loop_len = int(n/batch_size)

    A, B, C, D = ss  # state space model
    O_mat, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)  # data equation matrices
    H = np.hstack([O_mat, G])  # extended observation and impulse response matrix
    M = np.hstack(
        [np.zeros((regul_matrix.shape[0], O_mat.shape[1])), regul_matrix]
    )  # extended regularization matrix
    WtW = H.T @ H  # Normal equation
    Ht = H.T

    input_estimates = []

    for i in progressbar(range(loop_len), "Calculating estimates: ", loop_len, show_print=print_bar):
        if i == 0:
            batch = measurements[:bs,:]
        elif i == loop_len-1:
            batch = np.zeros((bs, measurements.shape[1]))  # last batch is zeros
        else:
            batch = measurements[i*batch_size-overlap:(i+1)*batch_size+overlap,:]

        Y = batch.reshape(-1,1)
        y = Ht @ Y  # Normal equation

        # init
        converged = False
        x_k = np.zeros((WtW.shape[0],1))
        r_k = y - WtW @ x_k
        if np.linalg.norm(r_k) < tol: converged = True
        p_k = r_k
        k = 0
        residuals = []

        # Conjugate gradient loop
        while not converged:
            a_k = (r_k.T @ r_k) / (p_k.T @ WtW @ p_k)
            x_k += a_k * p_k
            r_k_new = r_k - a_k * WtW @ p_k
            if np.linalg.norm(r_k_new) < tol: converged = True
            b_k = (r_k_new.T @ r_k_new) / (r_k.T @ r_k)
            p_k = r_k_new + b_k * p_k
            r_k = r_k_new
            residuals.append(np.linalg.norm(r_k))
            k += 1
            if k > max_iters: converged = True

        print("Number of iterations: ", k)

        input_estimates.append(x_k)

        if pickle_data:
            with open(fn + str(i) + ".pickle", 'wb') as handle:
                pickle.dump([estimate], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return input_estimates


def process_estimates(n_batches, overlap, estimates, nstates=0):
    """
    Here the input and initial state estimates are processed.
    Overlapped sections are discarded and the input estimate batches are stacked one after the other.
    """
    motor_estimates, propeller_estimates = [], []
    motor_est_overlap, prop_est_overlap = [], []
    for i in range(n_batches):
        if i == 0:
            all_motor_estimates = estimates[i][nstates::2]
            motor_est_overlap.append(all_motor_estimates)
            motor_estimates = all_motor_estimates[:-2*overlap]
            all_propeller_estimates = estimates[i][(nstates+1)::2]
            prop_est_overlap.append(all_propeller_estimates)
            propeller_estimates = all_propeller_estimates[:-2*overlap]
        else:
            all_motor_estimates = estimates[i][nstates::2]
            motor_est_overlap.append(all_motor_estimates)
            motor_estimates = np.concatenate(
                (motor_estimates, all_motor_estimates[overlap:-overlap])
            )
            all_propeller_estimates = estimates[i][(nstates+1)::2]
            prop_est_overlap.append(all_propeller_estimates)
            propeller_estimates = np.concatenate(
                (propeller_estimates, all_propeller_estimates[overlap:-overlap])
            )

    return motor_estimates, propeller_estimates


def trend_filter_cvx(ss, measurements, batch_size, overlap, times, lam=0.1, use_trend_filter=False, print_bar=True, pickle_data=False, fn='input_estimates'):
    """
    Analytical solution of the l2 regularized LS problem.
    Minimizes the sum of squared residuals, including an l2 constraint.
    """
    dt = np.mean(np.diff(times))
    n = len(times)
    bs = batch_size + 2*overlap
    loop_len = int(n/batch_size)

    A, B, C, D = ss  # state space model
    O_mat, G, D2, L, W = get_data_equation_matrices(A, B, C, D, n, bs)  # data equation matrices

    if use_trend_filter:
        regul_matrix = D2 # regularization matrix
    else:
        regul_matrix = L

    input_estimates = []

    for i in range(loop_len):
        if i == 0:
            batch = measurements[:bs,:]
        elif i == loop_len-1:
            batch = np.zeros((bs, measurements.shape[1]))
            # zero padding to finish estimation loop correctly
        else:
            batch = measurements[i*batch_size-overlap:(i+1)*batch_size+overlap,:]

        y = batch.reshape(-1,1)
        eps = 10

        # define optimization variables
        uhat_cvx = cp.Variable((regul_matrix.shape[1], 1), complex=False)

        # define objective function
        objective = cp.Minimize(
            cp.sum_squares(
                y - G @ uhat_cvx
            ) + lam * cp.sum_squares(regul_matrix @ uhat_cvx)
        )

        # define constraints
        constraints = [
            cp.abs( y - G @ uhat_cvx ) <= eps
        ]

        # define problem
        prob = cp.Problem(objective, constraints)

        # solve optimizaiton problem
        prob.solve()
        estimate = uhat_cvx.value
        print(i, estimate)
        input_estimates.append(estimate)  # estimate includes initial state estimate

        if pickle_data:
            with open(fn + str(i) + ".pickle", 'wb') as handle:
                pickle.dump([estimate], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return input_estimates
