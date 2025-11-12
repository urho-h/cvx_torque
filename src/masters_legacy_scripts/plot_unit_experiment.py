import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
from scipy.signal import dlsim

import testbench_MSSP as tb


plt.style.use('science')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 12,
    "figure.figsize": (6,4),
})


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def plot_batch(fn, n_batches):
    motor_data = np.loadtxt("../data/masters_data/processed_data/step_motor.csv", delimiter=",")
    time = motor_data[:,0]
    motor = motor_data[:,2]
    propeller = motor_data[:,-2]

    motor_estimates, propeller_estimates = [], []

    overlap = 50

    for i in range(n_batches):
        with open(fn + str(i) + ".pickle", 'rb') as handle:
            dataset = pickle.load(handle)
            if i == 0:
                all_motor_estimates = dataset[0][::2]
                motor_estimates = all_motor_estimates[:-2*overlap]
                all_propeller_estimates = dataset[0][1::2]
                propeller_estimates = all_propeller_estimates[:-2*overlap]
            else:
                all_motor_estimates = dataset[0][::2]
                motor_estimates = np.concatenate(
                    (motor_estimates, all_motor_estimates[overlap:-overlap])
                )
                all_propeller_estimates = dataset[0][1::2]
                propeller_estimates = np.concatenate(
                    (propeller_estimates, all_propeller_estimates[overlap:-overlap])
                )

    t_estimate = time[:len(propeller_estimates)]

    # plt.figure()
    # plt.plot(t_estimate, propeller_estimates, color='red', label='estimate')
    # plt.plot(time, propeller, color='blue', label='Propeller torque')
    # plt.legend()

    # plt.figure()
    # plt.plot(t_estimate, motor_estimates, color='red', label='estimate')
    # plt.plot(time, motor, color='blue', label='Motor torque')
    # plt.legend()

    #######################################
    sensor_data = np.loadtxt("../data/masters_data/processed_data/step_sensor.csv", delimiter=",")

    time = sensor_data[:,0]
    dt = np.mean(np.diff(time))

    U_est = np.hstack((motor_estimates, propeller_estimates))

    A, B, C, D = get_testbench_state_space(dt)
    C_mod = np.insert(C, C.shape[0], np.zeros((1, C.shape[1])), 0)
    C_mod[C.shape[0],22+18] += 2e4

    tout_ests, yout_ests, _ = dlsim(
        (A, B, C_mod, np.zeros((C_mod.shape[0], B.shape[1])), dt),
        U_est,
        t=t_estimate
    )

    with open(fn + "KF_0_mean.pickle", 'rb') as handle_kf:
        kf_dataset = pickle.load(handle_kf)
        time_kf = kf_dataset[0]
        input_kf = kf_dataset[1]
        torque_kf = kf_dataset[2]

    plt.figure()
    plt.plot(time, sensor_data[:,-1], label='Measurement', color='black')
    plt.plot(time_kf, torque_kf, color='royalblue')
    plt.plot(np.linspace(0, time[-1], yout_ests.shape[0]), yout_ests[:,-1], label='estimate', color='red')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_batch("estimates/kf_vs_hp/hp_trend_lam01_", 25)
