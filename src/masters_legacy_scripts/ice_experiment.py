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


def ice_excitation_data():
    times = np.genfromtxt("../data/ice_excitation/times.csv", delimiter=",")
    speeds = np.genfromtxt("../data/ice_excitation/speeds.csv", delimiter=",", usecols=(6,7,13,14,21))
    meas_speeds = np.genfromtxt("../data/ice_excitation/speed_measurements.csv", delimiter=",")
    torques = np.genfromtxt("../data/ice_excitation/torques.csv", delimiter=",", usecols=(8,18))
    meas_torques = np.genfromtxt("../data/ice_excitation/torque_measurements.csv", delimiter=",")
    motor = np.genfromtxt("../data/ice_excitation/motor.csv", delimiter=",")
    propeller = np.genfromtxt("../data/ice_excitation/propeller.csv", delimiter=",")

    return times, meas_speeds, meas_torques, torques, motor, propeller


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def ice_L_curve(times, dt, load, show_plot=False, pickle_data=False):
    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    tout, yout, _ = dlsim((A, B, C, D, dt), u=load, t=times)
    e3 = np.random.normal(0, .01, yout.shape)
    y_noise = yout + e3

    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 10, 100]

    l_norm, residual_norm = ie.L_curve(sys, y_noise, tout, lambdas, show_plot=True)

    if pickle_data:
        with open('estimates/ice_experiment_l_curve.pickle', 'wb') as handle:
            pickle.dump([l_norm, residual_norm], handle, protocol=pickle.HIGHEST_PROTOCOL)


def ice_excitation_simulated(run_cvx=False, run_kf=False, run_l_curve=False, show_plot=False, pickle_results=False):
    times, meas_speeds, meas_torques, torques, motor, propeller = ice_excitation_data()

    t_start = (times >= 6.6).nonzero()[0][0]
    t_end = (times <= 8.0).nonzero()[0][-1]

    one_hit = np.hstack((np.linspace(0, 0.34, 7), np.linspace(0.38, 0.64, 7), np.linspace(0.67, 0.87, 7), np.linspace(0.9, 0.98, 7)))
    ice_hit = np.hstack((one_hit, np.flip(one_hit)))
    ice_interaction = np.hstack((np.zeros(246), 1.5*ice_hit, 3.7*ice_hit, 6.0*ice_hit, 8.5*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 8.5*ice_hit, 6.0*ice_hit, 3.7*ice_hit, 1.5*ice_hit, np.zeros(257)))

    U_ice = np.zeros((len(ice_interaction), 2))

    e1 = np.random.normal(0, .01, U_ice.shape[0])
    e2 = np.random.normal(0, .01, U_ice.shape[0])

    U_ice[:,0] += 2.7 + e1
    U_ice[:,1] += ice_interaction + e2

    sim_times = times[t_start:t_end]-times[t_start]
    dt = np.mean(np.diff(sim_times))
    bs = int(len(sim_times)/3)

    plot_input = False
    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_ice[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.ylim(2.5,2.9)
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_ice[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.savefig("ice_setpoint.pdf")
        plt.show()

    if run_l_curve:
        ice_L_curve(sim_times, dt, U_ice, show_plot=True, pickle_data=True)

    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    C_mod = np.insert(C, 3, np.zeros((1, C.shape[1])), 0)
    C_mod[3,22+18] += 2e4
    D_mod = np.zeros((C_mod.shape[0], B.shape[1]))

    ice_to, ice_yo, _ = dlsim((A, B, C_mod, D_mod, dt), u=U_ice, t=sim_times)
    e3 = np.random.normal(0, .01, ice_yo.shape)
    y_noise = ice_yo + e3

    n = len(ice_to)
    ice_meas = y_noise
    motor = U_ice[:,0]
    propeller = U_ice[:,-1]

    C_one_speed = np.array(C[1:3,:])
    D_one_speed = np.zeros((C_one_speed.shape[0], B.shape[1]))
    sys = (A, B, C, D)

    vs = True

    if run_cvx:
        input_tikh, states_tikh = ie.estimate_input(
            sys,
            ice_meas[:,0:3],
            bs,
            ice_to,
            lam=0.05,
            use_virtual_sensor=vs
        )

        input_lasso, states_lasso = ie.estimate_input(
            sys,
            ice_meas[:,0:3],
            bs,
            ice_to,
            lam=0.05,
            use_zero_init=True,
            use_lasso=True,
            use_virtual_sensor=vs
        )

        if show_plot:
            ie.subplot_input_estimates(sim_times, motor, propeller, input_tikh, input_lasso, n, bs)

            if vs:
                ie.plot_virtual_sensor(sim_times[:-1], ice_yo[:-1,2:], states_tikh, states_lasso)

    if run_kf:
        times_kf, input_estimates_kf, torque_estimates_kf = kf.run_kalman_filter(
            sim_times,
            np.vstack((ice_meas[:,0], ice_meas[:,1])),
            ice_meas[:,2],
            ice_meas[:,2:],
            U_ice[:,0],
            U_ice[:,-1],
            np.mean(U_ice[:,0])
        )

    if pickle_results:
        with open('estimates/ice_experiment_simulated_lam005.pickle', 'wb') as handle:
            pickle.dump(
                [
                    sim_times,
                    U_ice[:,0],
                    U_ice[:,1],
                    input_tikh,
                    input_lasso,
                    ice_meas[:,2:],
                    states_tikh,
                    states_lasso,
                    input_estimates_kf,
                    torque_estimates_kf
                ],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)


def ice_experiment(run_cvx=False, run_kf=False, show_plot=False, pickle_results=False):
    times, meas_speeds, meas_torques, torques, motor, propeller = ice_excitation_data()

    t_start = (times >= 6.6).nonzero()[0][0]
    t_end = (times <= 8.0).nonzero()[0][-1]

    times = times[t_start:t_end]-times[t_start]
    dt = np.mean(np.diff(times))
    n = len(times)
    bs = int(len(times)/3)
    measurements = np.vstack((meas_speeds[:,t_start:t_end], meas_torques[t_start:t_end])).T

    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    vs = True

    if run_cvx:
        input_tikh, states_tikh = ie.estimate_input(
            sys,
            measurements,
            bs,
            times,
            lam=0.05,
            use_virtual_sensor=vs
        )

        input_lasso, states_lasso = ie.estimate_input(
            sys,
            measurements,
            bs,
            times,
            lam=0.05,
            use_lasso=True,
            use_virtual_sensor=vs
        )

        if show_plot:
            ie.subplot_input_estimates(times, motor[t_start:t_end], propeller[t_start:t_end], input_tikh, input_lasso, n, bs)

            if vs:
                ie.plot_virtual_sensor(times, torques[t_start:t_end], states_tikh, states_lasso)

    if run_kf:
        times_kf, input_estimates_kf, torque_estimates_kf = kf.run_kalman_filter(
            times,
            meas_speeds[:,t_start:t_end],
            meas_torques[t_start:t_end],
            torques[t_start:t_end],
            motor[t_start:t_end],
            propeller[t_start:t_end],
            2.7
        )

    if pickle_results:
        with open('estimates/ice_experiment_MSSP_measurements_lam005.pickle', 'wb') as handle:
            pickle.dump(
                [
                    times,
                    motor[t_start:t_end],
                    propeller[t_start:t_end],
                    input_tikh,
                    input_lasso,
                    torques[t_start:t_end],
                    states_tikh,
                    states_lasso,
                    input_estimates_kf,
                    torque_estimates_kf
                ],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ice_excitation_simulated(run_cvx=False, run_kf=False, run_l_curve=False, pickle_results=False)
    ice_experiment(run_cvx=False, run_kf=False, pickle_results=False)
