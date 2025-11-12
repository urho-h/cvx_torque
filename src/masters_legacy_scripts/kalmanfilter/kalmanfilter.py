import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim, butter, lfilter
from scipy.integrate import solve_ivp
import opentorsion as ot
import pickle

import sys
sys.path.append('../') # temporarily adds '../' to pythonpath so the drivetrain module is imported

import testbench_MSSP as tb


def ice_excitation_data():
    times = np.genfromtxt("../../data/ice_excitation/times.csv", delimiter=",")
    speeds = np.genfromtxt("../../data/ice_excitation/speeds.csv", delimiter=",", usecols=(6,7,13,14,21))
    meas_speeds = np.genfromtxt("../../data/ice_excitation/speed_measurements.csv", delimiter=",")
    torques = np.genfromtxt("../../data/ice_excitation/torques.csv", delimiter=",", usecols=(8,18))
    meas_torques = np.genfromtxt("../../data/ice_excitation/torque_measurements.csv", delimiter=",")
    motor = np.genfromtxt("../../data/ice_excitation/motor.csv", delimiter=",")
    propeller = np.genfromtxt("../../data/ice_excitation/propeller.csv", delimiter=",")

    return times, meas_speeds, meas_torques, torques, motor, propeller


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def W(sigma_m):
    """
    The input model in state-space form.
    """
    # var1, var2, a = 1, 0.1, 1 # design case 1
    var1, var2, a = 1, (10/3)**2, 1 # design case 2

    r0 = (1 + a**2) * var2 + var1
    r1 = -a * var2
    c = r0/2/r1 + np.sqrt((r0/2/r1)**2 - 1)
    sigma_e = np.sqrt(r1/c)

    W_A = np.array(([0, 0, 0], [0, a, c*sigma_e], [0, 0, 0]))
    W_B = np.array(([sigma_m, 0], [0, sigma_e], [0, 1]))
    W_C = np.array(([1, 0, 0], [0, 1, 0]))
    W_D = np.zeros((2, 2))

    return W_A, W_B, W_C, W_D


def extended_system(W_A, W_B, W_C, W_D, A, B, C, D, M, T, S, R, Q):
    """
    The extended state-space system used in the simultaneous input and state estimation.
    """
    nmasses = M.shape[0]
    Z1 = np.zeros((nmasses, nmasses-1))
    Z2 = np.zeros((nmasses-1, nmasses))
    Z3 = np.zeros((nmasses-1, nmasses-1))
    I = np.eye(nmasses)

    F = np.vstack([np.hstack([I, Z1]), np.hstack([Z2, Z3])])

    # the extended system
    Z_e = np.zeros((W_A.shape[0], A.shape[0]))

    A_e = np.vstack([np.hstack([W_A, Z_e]), np.hstack([(B @ W_C), A])])

    F_e = np.vstack([np.hstack([W_B, Z_e]), np.hstack([(B @ W_D), F])])

    C_e = np.hstack([np.zeros((C.shape[0], W_C.shape[1])), C])

    B_m = np.array(np.hstack([np.zeros((W_A.shape[0])), B[:,0]]))
    # convert B_m to nx1 array
    B_m = (np.zeros((1, B_m.shape[0])) + B_m).T

    T_e1 = np.hstack([W_C, np.zeros((2, A.shape[0]))])
    T_e2 = np.hstack([np.zeros((T.shape[0], W_A.shape[0])), T])
    T_e3 = np.hstack([np.zeros((nmasses, W_A.shape[0])), np.eye(nmasses), np.zeros((nmasses, nmasses-1))])
    T_e = np.vstack([T_e1, T_e2, T_e3])

    R_e = R

    Q_e1 = np.hstack([W_B @ W_B.T, W_B @ W_D.T @ B.T])
    Q_e21 = B @ W_D @ W_B.T
    Q_e22 = B @ W_D @ W_D.T @ B.T + F @ (F @ Q @ F.T) @ F.T
    Q_e2 = np.hstack([Q_e21, Q_e22])
    Q_e = np.vstack([Q_e1, Q_e2])

    S_e = np.vstack([np.eye(W_A.shape[0]), S])

    return A_e, F_e, C_e, B_m, T_e, R_e, Q_e, S_e


def fixed_lag_smoothing(A_e, F_e, C_e, B_m, T_e, R_e, Q_e, S_e, lag):
    A_fls1 = np.hstack([A_e, np.zeros((A_e.shape[0], (lag-1)*T_e.shape[0]))])
    A_fls2 = np.hstack([T_e, np.zeros((T_e.shape[0], (lag-1)*T_e.shape[0]))])
    A_fls3 = np.hstack([
        np.zeros(((lag-2)*T_e.shape[0], A_e.shape[0])),
        np.eye((lag-2)*T_e.shape[0]),
        np.zeros(((lag-2)*T_e.shape[0], T_e.shape[0]))
    ])
    A_fls = np.vstack([A_fls1, A_fls2, A_fls3])

    F_fls = np.vstack([F_e, np.zeros(((lag-1)*T_e.shape[0], F_e.shape[1]))])

    C_fls = np.hstack([C_e, np.zeros((C_e.shape[0], (lag-1)*T_e.shape[0]))])

    B_m_fls = np.vstack([B_m, np.zeros(((lag-1)*T_e.shape[0], 1))])

    T_fls = np.hstack([
        np.zeros((T_e.shape[0], A_e.shape[0])),
        np.zeros((T_e.shape[0], (lag-2)*T_e.shape[0])),
        np.eye(T_e.shape[0])
    ])

    R_fls = R_e

    Q_fls = LA.block_diag(Q_e, 0*np.eye((lag-1)*T_e.shape[0]))

    S_fls = S_e

    return A_fls, F_fls, C_fls, B_m_fls, T_fls, R_fls, Q_fls, S_fls


def construct_kalman_filter(A, C, T, Q, R, dt):
    P = LA.solve_discrete_are(A.T, C.T, Q, R) # ricatti_equation
    K = P @ C.T @ LA.inv(R + C @ P @ C.T)
    KC = K @ C
    Pz = T @ (np.eye((KC).shape[0]) - KC) @ P @ T.T

    KF_filter = dlti((np.eye(P.shape[0]) - K @ C) @ A, K, T, np.zeros((T.shape[0], C.shape[0])), dt=dt)
    KF_pred = dlti(A - K @ C, K, T, np.zeros((T.shape[0], C.shape[0])), dt=dt)

    return KF_filter, KF_pred, Pz, K


def run_kalman_filter(times, meas_speeds, meas_torques, torques, motor, propeller, mu_m):
    dt = np.mean(np.diff(times))

    Fs = 1000 # Sampling frequency
    sigma_m = 2/3 # worst case variance

    ## Sensor params
    rpm_sensor_locations = [6, 7]
    torque_sensor_locations = [8]
    n_sensors = len(rpm_sensor_locations) + len(torque_sensor_locations)

    stiffnesses = np.array(
        [1.90e5,
         6.95e3,
         90.00,
         90.00,
         90.00,
         90.00,
         30.13,
         4.19e4,
         5.40e3,
         4.19e4,
         1.22e3,
         4.33e4,
         3.10e4,
         1.14e3,
         3.10e4,
         1.22e4,
         4.43e4,
         1.38e5,
         2.00e4,
         1.38e5,
         1.22e4]
    )

    ## KALMAN FILTER DESIGN PARAMETERS
    lag = 2 #10
    R = np.diag([0.05, 0.1, 0.2]) # measurement covariance R = E{v*v'}
    Q = 0.01*np.eye(43) # diagonal matrix with shape of the A matrix

    A, B, C, D = get_testbench_state_space(dt)

    M = np.zeros((22, 22))
    T = np.hstack([np.zeros((M.shape[0]-1, M.shape[0]-1)), np.zeros((M.shape[0]-1, 1)), np.diag(stiffnesses)]) # a matrix defining which states to be estimated
    S = np.zeros((C.T).shape)

    W_A, W_B, W_C, W_D = W(sigma_m)

    A_e, F_e, C_e, B_m, T_e, R_e, Q_e, S_e = extended_system(W_A, W_B, W_C, W_D, A, B, C, D, M, T, S, R, Q)

    A_fls, F_fls, C_fls, B_m_fls, T_fls, R_fls, Q_fls, S_fls = fixed_lag_smoothing(A_e, F_e, C_e, B_m, T_e, R_e, Q_e, S_e, lag)

    KF_filter, KF_pred, Pz, K = construct_kalman_filter(A_fls, C_fls, T_fls, Q_fls, R_fls, dt)
    # KF_filter, KF_pred, Pz, K = construct_kalman_filter(A_e, C_e, T_e, Q_e, R_e, dt)

    ## Estimation
    y = np.vstack([meas_speeds, meas_torques])
    u = mu_m*np.ones(y.shape[1])
    Y = np.vstack([u, y])

    KF2 = dlti(
        KF_filter.A,
        np.hstack([B_m_fls, KF_filter.B]),
        KF_filter.C,
        np.zeros((KF_filter.C.shape[0], KF_filter.B.shape[1]+1)),
        dt=dt
    )

    # KF2 = dlti(
    #     KF_filter.A,
    #     np.hstack([B_m, KF_filter.B]),
    #     KF_filter.C,
    #     np.zeros((KF_filter.C.shape[0], KF_filter.B.shape[1]+1)),
    #     dt=dt
    # )

    tout, yout, xout = dlsim(KF2, Y.T, t=times) # z_hat

    input_estimates = yout[:,:2]
    torque_estimates = yout[:,2:22]
    speed_estimates = yout[:,22+2:]

    plt.figure()
    plt.plot(times, meas_torques, label="measured")
    plt.plot(times[:-lag], torque_estimates[lag:,rpm_sensor_locations[-1]], label="estimate", alpha=0.5)
    # plt.xlim(6.5, 8)
    # plt.ylim(0, 4)
    plt.title('Torque transducer 1')
    plt.legend()

    plt.subplot(211)
    plt.plot(times[100:], torques[100:,-1], label="Measured")
    plt.plot(times[100:-lag], torque_estimates[lag+100:,rpm_sensor_locations[-1]+10], label="Estimate", linestyle='dashed')
    plt.title('Torque transducer 2')
    # plt.ylim(0, 30)
    plt.legend()
    plt.grid()

    plt.subplot(212)
    plt.plot(times[100:], input_estimates[100:], label=("Motor side estimate", "Propeller side estimate"))
    plt.title('Input estimates')
    plt.legend()
    plt.grid()
    # plt.savefig("kf_step_estimates.pdf")

    plt.show()

    return times, input_estimates, torque_estimates


if __name__ == "__main__":
    mu_m = 2.7 # mean
    times, meas_speeds, meas_torques, torques, motor, propeller = ice_excitation_data()
    run_kalman_filter(times, meas_speeds, meas_torques, torques, motor, propeller, mu_m)
