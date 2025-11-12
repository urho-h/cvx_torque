import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlsim

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

    objective = cp.Minimize(cp.sum_squares(measurements - obsrv @ x - gamm @ d) + lam1 * cp.sum_squares(regu @ d) + lam2 * cp.pnorm(regu @ d, 1))

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

    return O, G, D2, L


def progressbar(it, prefix="", size=60, out=sys.stdout):
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
    print("\n", flush=True, file=out)


def L_curve(sys, measurements, times, lambdas, use_zero_init=True):
    dt = np.mean(np.diff(times))
    bs = len(times)
    n = len(times)

    A, B, C, D = sys
    O, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_zero_init:
        x_init = np.zeros((O.shape[1], 1))
    else:
        x_init = None

    input_estimates = []

    y = measurements.reshape(-1,1)

    for i in progressbar(range(len(lambdas)), "Calculating estimates :", len(lambdas)):
        estimate, x_init = tikhonov_problem(y, O, G, D2, initial_state=x_init, lam=lambdas[i])
        input_estimates.append(estimate)

    norm, res_norm = [], []
    for i in range(len(lambdas)):
        norm.append(np.linalg.norm(y - G @ input_estimates[i]))
        res_norm.append(np.linalg.norm(D2 @ input_estimates[i]))

    return norm, res_norm


def pareto_curve(sys, measurements, times, lambdas, use_zero_init=True):
    dt = np.mean(np.diff(times))
    bs = len(times)
    n = len(times)

    A, B, C, D = sys
    O, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_zero_init:
        x_init = np.zeros((O.shape[1], 1))
    else:
        x_init = None

    input_estimates = []

    y = measurements.reshape(-1,1)

    for i in progressbar(range(len(lambdas)), "Calculating estimates :", len(lambdas)):
        estimate, x_init = lasso_problem(y, O, G, L, initial_state=x_init, lam=lambdas[i])
        input_estimates.append(estimate)

    for i in range(len(lambdas)):
        # plt.yscale("log")
        # plt.xscale("log")
        plt.scatter(np.linalg.norm(L @ input_estimates[i], ord=1), np.linalg.norm(y - G @ input_estimates[i]), color='blue')

    plt.ylabel("$||y-\Gamma u||$")
    plt.xlabel("$||L u||$")
    plt.show()


def data_eq_simulation(sys, times, load, bs):
    dt = np.mean(np.diff(times))
    n = len(times)

    A, B, C, D = sys
    # omat = de.O(A, C, bs)
    gmat = de.gamma(A, B, C, bs)

    u = load.T.reshape(-1,1)
    print(u.shape)
    print(gmat.shape)

    return gmat @ u


def estimate_input(sys, measurements, bs, times, lam=0.1, use_zero_init=True, use_lasso=False, use_elastic_net=False, use_trend_filter=False, use_virtual_sensor=False):
    dt = np.mean(np.diff(times))
    n = len(times)
    loop_len = int(n/bs)

    A, B, C, D = sys
    O, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_trend_filter:
        regul_matrix = D2
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
        batch = measurements[i*bs:(i+1)*bs,:]
        y = batch.reshape(-1,1)

        if use_lasso:
            estimate, x_init = lasso_problem(y, O, G, regul_matrix, initial_state=x_init, lam=lam)
        elif use_elastic_net:
            estimate, x_init = elastic_net_problem(y, O, G, regul_matrix, initial_state=x_init, lam=lam)
        else:
            estimate, x_init = tikhonov_problem(y, O, G, regul_matrix, initial_state=x_init, lam=lam)

        x_est = omat @ x_init + gmat @ estimate
        x_init = x_est[-A.shape[0]:,:]

        input_estimates.append(estimate)

    yout_ests = None

    if use_virtual_sensor:
        ests = input_estimates[0]
        for i in progressbar(range(1, loop_len), "Calculating virtual sensor result: ", loop_len):
            ests = np.vstack((ests, input_estimates[i]))

        motor_est = ests[::2]
        propeller_est = ests[1::2]

        U_est = np.hstack((motor_est, propeller_est))

        C_mod = np.insert(C, C.shape[0], np.zeros((1, C.shape[1])), 0)
        C_mod[C.shape[0],22+18] += 2e4

        tout_ests, yout_ests, _ = dlsim(
            (A, B, C_mod, np.zeros((C_mod.shape[0], B.shape[1])), dt),
            U_est,
            t=times[:U_est.shape[0]]
        )

    return input_estimates, yout_ests


def subplot_input_estimates(time, tau_motor, tau_propeller, tikh_estimates, lasso_estimates, n, bs):
    loop_len = int(n/bs)
    ax1 = plt.subplot(211)
    for i in range(loop_len):
        plt.plot(time[i*bs:(i+1)*bs], tau_motor[i*bs:(i+1)*bs], linestyle='solid', color='blue')
        plt.plot(time[i*bs:(i+1)*bs], tikh_estimates[i][::2], linestyle='solid', color='red')
        plt.plot(time[i*bs:(i+1)*bs], lasso_estimates[i][::2], linestyle='solid', color='red')

    plt.legend(('Motor side input', 'H-P trend', '$\ell_1$ trend'))
    plt.legend(('Motor side input', '$\ell_1$ trend'))
    plt.ylabel('Torque (Nm)')
    plt.xlabel('Time (s)')
    # plt.xlim(0.1, 10)
    # plt.ylim(-3, 3)
    plt.grid()
    # plt.tick_params('x', labelbottom=False)

    ax2 = plt.subplot(212)
    for i in range(loop_len):
        plt.plot(time[i*bs:(i+1)*bs], tau_propeller[i*bs:(i+1)*bs], linestyle='solid', color='C0')
        plt.plot(time[i*bs:(i+1)*bs], tikh_estimates[i][1::2], linestyle='solid', color='C1')
        plt.plot(time[i*bs:(i+1)*bs], lasso_estimates[i][1::2], linestyle='solid', color='C2')

    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    # plt.xlim(0.1, 10)
    # plt.ylim(-6, 4)
    plt.grid()
    # plt.savefig("ramp_input_estimate.pdf")

    plt.show()


def plot_input_estimates(time, tau_motor, tau_propeller, tikh_estimates, lasso_estimates, n, bs):
    loop_len = int(n/bs)
    plt.figure()
    for i in range(loop_len):
        plt.plot(time[i*bs:(i+1)*bs], tau_motor[i*bs:(i+1)*bs], linestyle='solid', color='black')
        plt.plot(time[i*bs:(i+1)*bs], tikh_estimates[i][::2], linestyle='dotted', color='red')
        plt.plot(time[i*bs:(i+1)*bs], lasso_estimates[i][::2], linestyle='dashed', alpha=0.3, color='blue')

    plt.legend(('known input', 'Tikhonov estimate', 'LASSO estimate'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Motor side input estimates (H-P trend filtering)')

    plt.figure()
    for i in range(loop_len):
        plt.plot(time[i*bs:(i+1)*bs], tau_propeller[i*bs:(i+1)*bs], linestyle='solid', color='black')
        plt.plot(time[i*bs:(i+1)*bs], tikh_estimates[i][1::2], linestyle='dotted', color='red')
        plt.plot(time[i*bs:(i+1)*bs], lasso_estimates[i][1::2], linestyle='dashed', alpha=0.3, color='blue')

    plt.legend(('known input', 'Tikhonov estimate', 'LASSO estimate'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Propeller side input estimates (H-P trend filtering)')

    plt.show()


def plot_virtual_sensor(times, torques, yout_tikh, yout_lasso):
    plt.figure()
    plt.plot(times[100:], torques[100:,-2], label='Measured', linestyle='solid', color='blue')
    plt.plot(times[100:], yout_tikh[100:,-2], label='Tikhonov estimate', linestyle='solid', color='red')
    plt.plot(times[100:], yout_lasso[100:,-2], label='LASSO estimate', alpha=0.5, linestyle='solid', color='green')
    plt.legend()
    plt.title('Torque transducer 1')

    plt.figure(figsize=(12,4))
    plt.plot(times[100:], torques[100:,-1], label='Measured torque', linestyle='solid', color='blue')
    plt.plot(times[100:], yout_tikh[100:,-1], label='H-P trend', linestyle='solid', color='red')
    plt.plot(times[100:], yout_lasso[100:,-1], label='$\ell_1$ trend', alpha=0.5, linestyle='solid', color='green')
    plt.legend()
    plt.title('Torque transducer 2')
    plt.grid()

    # plt.ylim(0, 25)
    # plt.xlim(6.5, 8)
    plt.ylabel('Torque (Nm)')
    plt.xlabel('Time (s)')
    # plt.savefig("ramp_torque_transducer.pdf")

    plt.show()
