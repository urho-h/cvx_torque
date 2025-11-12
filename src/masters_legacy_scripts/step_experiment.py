import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import lsim, dlsim, butter, lfilter
import pickle

import testbench_MSSP as tb
from convex_optimization import input_estimation as ie
from kalmanfilter import kalmanfilter as kf


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 12,
    "figure.figsize": (6,4),
})


def low_pass_filter(signal, cutoff, fs):
    '''
    A fifth-order Butterworth low-pass filter.
    '''
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(5, normalized_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def step_L_curve(times, dt, load, show_plot=False, pickle_data=False):
    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    tout, yout, _ = dlsim((A, B, C, D, dt), u=load, t=times)
    e3 = np.random.normal(0, .01, yout.shape)
    y_noise = yout + e3

    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 10, 100]

    l_norm, residual_norm = ie.L_curve(sys, y_noise, tout, lambdas, show_plot=True)

    if pickle_data:
        with open('estimates/step_experiment_l_curve.pickle', 'wb') as handle:
            pickle.dump([l_norm, residual_norm], handle, protocol=pickle.HIGHEST_PROTOCOL)


def step_experiment(run_cvx=False, run_kf=False, run_l_curve=False, pickle_results=False):
    fs = 1000
    sim_times = np.arange(0, 20, 1/fs)
    dt = np.mean(np.diff(sim_times))
    bs = 500

    U_step = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_step.shape[0])
    e2 = np.random.normal(0, .01, U_step.shape[0])

    # U_step[:,0] += 2.5 + e1
    U_step[10000:,0] += np.flip(np.linspace(-2.5, 2.5, 10000)) + e1[10000:]
    U_step[:,1] += e2
    U_step[13200:15200,1] += 1.0
    U_step[15200:18200,1] -= 4.0
    U_step[18200:,1] += 0.5

    plot_input = False
    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times[:10000], U_step[10000:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times[:10000], U_step[10000:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.savefig("step_setpoint.pdf")
        plt.show(sim_times, dt, U_step, show_plot=True, pickle_data=True)

    if run_l_curve:
        step_L_curve(sim_times[:1000], dt, U_step[13000:14000], show_plot=True, pickle_data=True)

    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    C_mod = np.insert(C, 3, np.zeros((1, C.shape[1])), 0)
    C_mod[3,22+18] += 2e4
    D_mod = np.zeros((C_mod.shape[0], B.shape[1]))

    step_to, step_yo, _ = dlsim((A, B, C_mod, D_mod, dt), u=U_step, t=sim_times)
    e3 = np.random.normal(0, .01, step_yo.shape)
    y_noise = step_yo + e3

    n = len(step_to[10000:])
    step_meas = y_noise[10000:]
    motor = U_step[10000:,0]
    propeller = U_step[10000:,-1]

    C_one_speed = np.array(C[1:3,:])
    D_one_speed = np.zeros((C_one_speed.shape[0], B.shape[1]))
    sys = (A, B, C, D)

    vs = True

    if run_cvx:
        input_tikh, states_tikh = ie.estimate_input(
            sys,
            step_meas[:,0:3],
            bs,
            step_to[:10000],
            lam=1,
            use_virtual_sensor=vs
        )

        input_lasso, states_lasso = ie.estimate_input(
            sys,
            step_meas[:,0:3],
            bs,
            step_to[:10000],
            lam=0.001,
            use_zero_init=True,
            use_lasso=True,
            use_virtual_sensor=vs
        )

        ie.subplot_input_estimates(sim_times, motor, propeller, input_tikh, input_lasso, n, bs)

        if vs:
            ie.plot_virtual_sensor(sim_times[:10000], step_yo[10000:,2:], states_tikh, states_lasso)

    if run_kf:
        times_kf, input_estimates_kf, torque_estimates_kf = kf.run_kalman_filter(
            sim_times[:10000],
            np.vstack((step_meas[:,0], step_meas[:,1])),
            step_meas[:,2],
            step_meas[:,2:],
            U_step[:,0],
            U_step[:,-1],
            np.mean(U_step[:,0])
        )

    if pickle_results:
        with open('estimates/step_experiment_lam1_lam0001.pickle', 'wb') as handle:
            pickle.dump(
                [
                    sim_times,
                    U_step[:,0],
                    U_step[:,1],
                    input_tikh,
                    input_lasso,
                    step_meas[:,2:],
                    states_tikh,
                    states_lasso,
                    input_estimates_kf,
                    torque_estimates_kf
                ],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)


def sparse_experiment(run_cvx=False, run_kf=False):
    fs = 1000
    sim_times = np.arange(0, 20, 1/fs)
    dt = np.mean(np.diff(sim_times))
    bs = 500

    U_sparse = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_sparse.shape[0])
    e2 = np.random.normal(0, .01, U_sparse.shape[0])

    U_sparse[:,0] += 2.5 + e1
    U_sparse[:,1] += e2
    U_sparse[13210:13220,1] += np.arange(0, 10, 1)
    U_sparse[13220:13230,1] += np.flip(np.arange(0, 10, 1))
    U_sparse[15210:15220,1] -= np.arange(0, 10, 1)
    U_sparse[15220:15230,1] -= np.flip(np.arange(0, 10, 1))

    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    C_mod = np.insert(C, 3, np.zeros((1, C.shape[1])), 0)
    C_mod[3,22+18] += 2e4
    D_mod = np.zeros((C_mod.shape[0], B.shape[1]))

    sparse_to, sparse_yo, _ = dlsim((A, B, C_mod, D_mod, dt), u=U_sparse, t=sim_times)
    e3 = np.random.normal(0, .01, sparse_yo.shape)
    y_noise = sparse_yo + e3

    n = len(sparse_to[10000:])
    sparse_meas = y_noise[10000:]
    motor = U_sparse[10000:,0]
    propeller = U_sparse[10000:,-1]

    C_one_speed = np.array(C[1:3,:])
    D_one_speed = np.zeros((C_one_speed.shape[0], B.shape[1]))
    # sys = (A, B, C_one_speed, D_one_speed)
    sys = (A, B, C, D)

    vs = True

    if run_cvx:
        input_tikh, states_tikh = ie.estimate_input(
            sys,
            sparse_meas[:,0:3],
            bs,
            sparse_to[:10000],
            lam=8,
            use_virtual_sensor=vs
        )

        input_lasso, states_lasso = ie.estimate_input(
            sys,
            sparse_meas[:,0:3],
            bs,
            sparse_to[:10000],
            lam=8,
            use_lasso=True,
            use_virtual_sensor=vs
        )

        # ie.plot_input_estimates(sim_times, motor, propeller, input_tikh, input_lasso, n, bs)
        ie.subplot_input_estimates(sim_times, motor, propeller, input_tikh, input_lasso, n, bs)

        if vs:
            ie.plot_virtual_sensor(sim_times[:10000], y_noise[10000:,2:], states_tikh, states_lasso)

    if run_kf:
        _, _, _ = kf.run_kalman_filter(
            sim_times[:10000],
            np.vstack((sparse_meas[:,0], sparse_meas[:,1])),
            sparse_meas[:,2],
            sparse_meas[:,2:],
            U_sparse[:,0],
            U_sparse[:,-1],
            np.mean(U_sparse[:,0])
        )


def sparse_L_curve_test():
    fs = 1000
    sim_times = np.arange(0, 0.5, 1/fs)
    dt = np.mean(np.diff(sim_times))

    U_sparse = np.zeros((len(sim_times), 2))

    U_sparse[:,0] += 2.5
    U_sparse[200:220,1] += np.arange(0, 2, 1/10)
    U_sparse[220:240,1] += np.flip(np.arange(0, 2, 1/10))

    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    sparse_to, sparse_yo, _ = dlsim((A, B, C, D, dt), u=U_sparse, t=sim_times)
    e3 = np.random.normal(0, .01, sparse_yo.shape)
    y_noise = sparse_yo + e3

    # lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    lambdas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    ie.L_curve(sys, y_noise, sparse_to, lambdas)


def sparse_pareto_curve_test():
    fs = 1000
    sim_times = np.arange(0, 0.5, 1/fs)
    dt = np.mean(np.diff(sim_times))

    U_sparse = np.zeros((len(sim_times), 2))

    U_sparse[:,0] += 2.5
    U_sparse[200:220,1] += np.arange(0, 2, 1/10)
    U_sparse[220:240,1] += np.flip(np.arange(0, 2, 1/10))

    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    sparse_to, sparse_yo, _ = dlsim((A, B, C, D, dt), u=U_sparse, t=sim_times)
    e3 = np.random.normal(0, .01, sparse_yo.shape)
    y_noise = sparse_yo + e3

    # lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    ie.pareto_curve(sys, y_noise, sparse_to, lambdas)


def reversal_experiment():

    def reversal(start, end, n):
        """
        Generates a step signal with an initial value of `start` and an end value of `end`.

        Args:
        start (float): Initial value of the step signal.
        end (float): End value of the step signal.
        n (int): Length of the step signal to generate.

        Returns:
        numpy.ndarray: Step signal of length n.
        """
        # Initialize the output signal
        signal = np.zeros(n)
        # Set the initial value
        signal[0] = start
        # Calculate the step size
        step = (end - start) / (n - 1)
        # Generate the step signal
        for i in range(1, n):
            signal[i] = signal[i-1] + step

        return signal

    reversal_signal = reversal(200, -200, len(sim_times))

    return


def sinusoidal_experiment():

    def sum_sines(freqs, amps, phases, times, dc_offset):
        """
        Generates a sinusoidal signal that is a sum of three sine waves with different frequencies.

        Args:
        freqs (list[float]): List of three frequencies for the sine waves.
        amps (list[float]): List of three amplitudes for the sine waves.
        phases (list[float]): List of three phase shifts for the sine waves.
        times (numpy.ndarray): Timesteps of the signal in seconds.

        Returns:
        numpy.ndarray: Sinusoidal signal that is a sum of three sine waves.
        """

        signal = np.zeros(len(times))
        for f, a, p in zip(freqs, amps, phases):
            signal += a * np.sin(2 * np.pi * f * times + p)

        return signal + dc_offset

    fs = 1000
    sim_times = np.arange(0, 20, 1/fs)
    dt = np.mean(np.diff(sim_times))
    bs = 500

    freqs = [20, 40, 60]
    amps = [2, 1, 0.5]
    phases = [0, 0, 0]
    offset = 0
    sine_signal = sum_sines(freqs, amps, phases, sim_times, offset)

    plt.plot(sim_times, sine_signal)
    plt.show()

    return


if __name__ == "__main__":
    step_experiment(run_cvx=False, run_kf=False, run_l_curve=False, pickle_results=False)

    # sparse_L_curve_test()
    # sparse_pareto_curve_test()
    # sparse_experiment(run_cvx=True, run_kf=False)
