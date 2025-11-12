import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
from scipy.signal import dlsim
from matplotlib import gridspec

import testbench_MSSP as tb
import simulation_pareto_curve as spc


plt.style.use('science')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 12,
    "figure.figsize": (8,5), #(7,6)
})


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def get_states(fn, n_batches, case, sensor_data):
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

    time = sensor_data[:,0]
    dt = np.mean(np.diff(time))
    t_estimate = np.linspace(0, len(propeller_estimates)*dt, len(propeller_estimates))

    U_est = np.hstack((motor_estimates, propeller_estimates))

    A, B, C, D = get_testbench_state_space(dt)
    C_mod = np.insert(C, C.shape[0], np.zeros((1, C.shape[1])), 0)
    C_mod[C.shape[0],22+18] += 2e4

    tout_ests, yout_ests, _ = dlsim(
        (A, B, C_mod, np.zeros((C_mod.shape[0], B.shape[1])), dt),
        U_est,
        t=t_estimate
    )

    return t_estimate, yout_ests[:-1,-1]


def plot_ice_different_lambda():
    case = "ice_2000"
    n = 20
    sensor_data_ice = np.loadtxt("../data/masters_data/processed_data/" + case + "_sensor.csv", delimiter=",")

    tehp_00001, dehp_00001 = get_states("estimates/" + case + "_experiment/hp_different_lambdas/hp_trend_lam00001_", n, case, sensor_data_ice)
    # tehp_001, dehp_001 = get_states("estimates/" + case + "_experiment/hp_different_lambdas/hp_trend_lam001_", n, case, sensor_data_ice)
    tehp_01, dehp_01 = get_states("estimates/" + case + "_experiment/hp_trend_lam01_", n, case, sensor_data_ice)
    tehp_10, dehp_10 = get_states("estimates/" + case + "_experiment/hp_different_lambdas/hp_trend_lam10_", n, case, sensor_data_ice)
    tehp_100, dehp_100 = get_states("estimates/" + case + "_experiment/hp_different_lambdas/hp_trend_lam100_", n, case, sensor_data_ice)
    tehp_1000, dehp_1000 = get_states("estimates/" + case + "_experiment/hp_different_lambdas/hp_trend_lam1000_", n, case, sensor_data_ice)

    sim_times, impulse_load, sinusoidal_load, step_load, ramp_load = spc.get_unit_test_loads()
    with open("estimates/pareto_curves_new/impulse_experiment_hp_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_hp_trend = dataset[0]
        residual_norm_hp_trend = dataset[1]
        lambdas_hp_trend = dataset[2]

    plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,2])
    a1 = plt.subplot(gs[0])
    plt.title("a)",loc='left')
    a1.set_yscale("log")
    a1.set_xscale("log")
    a1.scatter(residual_norm_hp_trend, l_norm_hp_trend, color='black')
    a1.scatter(residual_norm_hp_trend[3], l_norm_hp_trend[3], color='C0')
    a1.scatter(residual_norm_hp_trend[6], l_norm_hp_trend[6], color='C3')
    a1.scatter(residual_norm_hp_trend[8], l_norm_hp_trend[8], color='C1')
    a1.scatter(residual_norm_hp_trend[9], l_norm_hp_trend[9], color='C2')
    a1.scatter(residual_norm_hp_trend[10], l_norm_hp_trend[10], color='C4')
    a1.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[3]),
        (residual_norm_hp_trend[3], l_norm_hp_trend[3]),
        (residual_norm_hp_trend[3], l_norm_hp_trend[3]+300)
    )
    a1.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[6]),
        (residual_norm_hp_trend[6], l_norm_hp_trend[6]),
        (residual_norm_hp_trend[6]-0.6, l_norm_hp_trend[6]-0.5)
    )
    a1.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[8]),
        (residual_norm_hp_trend[8], l_norm_hp_trend[8]),
        (residual_norm_hp_trend[8], l_norm_hp_trend[8]+0.5)
    )
    a1.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[9]),
        (residual_norm_hp_trend[9], l_norm_hp_trend[9]),
        (residual_norm_hp_trend[9], l_norm_hp_trend[9]+0.1)
    )
    a1.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[10]),
        (residual_norm_hp_trend[10], l_norm_hp_trend[10]),
        (residual_norm_hp_trend[10]-2, l_norm_hp_trend[10]-0.001)
    )
    a1.set_ylabel("$||L u||_2$")
    a1.set_xlabel("$||y-\Gamma u||_2$")
    # a1.set_xticks(ticks=[3,4,6,10], labels=[None,None,None,"$10^1$"])

    a0 = plt.subplot(gs[1])
    plt.title("b)",loc='left')
    a0.plot(sensor_data_ice[:,0], sensor_data_ice[:,-1], color='black', label='Measurement')
    a0.plot(tehp_00001, dehp_00001, color='C0', label='$\lambda$ = 0.0001')
    # plt.plot(tehp_001, dehp_001, label='$\lambda$ = 0.01')
    a0.plot(tehp_01, dehp_01, color='C3', label='$\lambda$ = 0.1')
    a0.plot(tehp_10, dehp_10, color='C1', label='$\lambda$ = 10')
    a0.plot(tehp_100, dehp_100, color='C2', label='$\lambda$ = 100')
    a0.plot(tehp_1000, dehp_1000, color='C4', label='$\lambda$ = 1000')
    a0.set_xlim(4.375,4.435)
    a0.set_ylim(2,24.5)
    a0.legend()
    a0.set_ylabel("Torque (Nm)")
    a0.set_xlabel("Time (s)")
    plt.savefig("../figures/ice_2000_diff_lambda.pdf")
    # plt.tight_layout()
    plt.show()


def plot_maritime_estimates():
    case = "experiments_redone/ice"
    # n = 32
    n = 20
    sensor_data_ice = np.loadtxt("../data/masters_data/processed_data/ice_2000_sensor.csv", delimiter=",")

    tet, det = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_ice)
    tel1, del1 = get_states("estimates/" + case + "_experiment/l1_lam1_", n, case, sensor_data_ice)
    tehp, dehp = get_states("estimates/" + case + "_experiment/hp_trend_lam01_", n, case, sensor_data_ice)
    tel1t, del1t = get_states("estimates/" + case + "_experiment/l1_trend_lam1_", n, case, sensor_data_ice)

    plt.figure()
    ax1 = plt.subplot(411)
    ax1.plot(sensor_data_ice[:,0], sensor_data_ice[:,-1], color='black', label='Measurement')
    ax1.plot(tet, det, color='red', linestyle='dashed', label='Tikhonov regularization')
    ax1.plot([100,100], [100,100], color='blue', linestyle='dashed', label='$\ell_1$ regularization')
    ax1.plot([100,100], [100,100], color='red', label='H-P trend filtering')
    ax1.plot([100,100], [100,100], color='blue', label='$\ell_1$ trend filtering')
    ax1.set_xlim(3.8,5)
    ax1.set_ylim(2,24)
    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.6),
        fancybox=True,
        shadow=True,
        ncol=3
    )
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    plt.subplot(412)
    plt.plot(sensor_data_ice[:,0], sensor_data_ice[:,-1], color='black', label='Measurement')
    plt.plot(tel1, del1, color='blue', linestyle='dashed', label='$\ell_1$ regularization')
    plt.xlim(3.8,5)
    plt.ylim(2,24)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    plt.subplot(413)
    plt.plot(sensor_data_ice[:,0], sensor_data_ice[:,-1], color='black', label='Measurement')
    plt.plot(tehp, dehp, color='red', label='H-P trend filtering')
    plt.xlim(3.8,5)
    plt.ylim(2,24)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    plt.subplot(414)
    plt.plot(sensor_data_ice[:,0], sensor_data_ice[:,-1], color='black', label='Measurement')
    plt.plot(tel1t, del1t, color='blue', label='$\ell_1$ trend filtering')
    # plt.xlim(7.4,8.7)
    plt.xlim(3.8,5)
    plt.ylim(2,24)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")
    # plt.savefig("../figures/ice_2000_torque_estimate_all_subplots.pdf")
    # plt.show()

    case = "experiments_redone/CFD"
    # n = 71
    n = 76
    sensor_data_CFD = np.loadtxt("../data/masters_data/processed_data/CFD_2000_sensor.csv", delimiter=",")

    tet, det = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_CFD)
    tel1, del1 = get_states("estimates/" + case + "_experiment/l1_lam001_", n, case, sensor_data_CFD)
    tehp, dehp = get_states("estimates/" + case + "_experiment/hp_trend_lam10_", n, case, sensor_data_CFD)
    tel1t, del1t = get_states("estimates/" + case + "_experiment/l1_trend_lam10_", n, case, sensor_data_CFD)

    plt.figure()
    plt.subplot(211)
    plt.title("a)",loc='left')
    plt.plot(sensor_data_CFD[:,0], sensor_data_CFD[:,-1], color='black', label='Measurement')
    plt.plot(tet, det, color='red', linestyle='dashed', label='Tikhonov regularization')
    plt.plot(tel1, del1, color='blue', linestyle='dashed', label='$\ell_1$ regularization', alpha=0.7)
    plt.legend()
    plt.xlim(3,32)
    plt.ylim(0,22)
    plt.ylabel("Torque (Nm)")

    plt.subplot(212)
    plt.plot(sensor_data_CFD[:,0], sensor_data_CFD[:,-1], color='black', label='Measurement')
    plt.plot(tehp, dehp, color='red', label='H-P trend filtering')
    plt.plot(tel1t, del1t, color='blue', label='$\ell_1$ trend filtering')
    plt.legend()
    plt.xlim(3,32)
    plt.ylim(0,22)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")
    plt.savefig("../figures/CFD_2000_torque_estimate_all.pdf")
    # plt.show()

    plt.figure()
    plt.subplot(211)
    plt.title("b)",loc='left')
    plt.plot(sensor_data_CFD[:,0], sensor_data_CFD[:,-1], color='black', label='Measurement')
    plt.plot(tet, det, color='red', linestyle='dashed', label='Tikhonov regularization')
    plt.plot(tel1, del1, color='blue', linestyle='dashed', label='$\ell_1$ regularization', alpha=0.7)
    plt.legend()
    plt.xlim(17.51,18)
    plt.ylim(-3,21)
    plt.ylabel("Torque (Nm)")

    plt.subplot(212)
    plt.plot(sensor_data_CFD[:,0], sensor_data_CFD[:,-1], color='black', label='Measurement')
    plt.plot(tehp, dehp, color='red', label='H-P trend filtering')
    plt.plot(tel1t, del1t, color='blue', label='$\ell_1$ trend filtering')
    plt.legend()
    plt.xlim(17.51,18)
    plt.ylim(-3,21)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")
    plt.savefig("../figures/CFD_2000_torque_estimate_all_zoomed.pdf")
    plt.show()


def plot_unit_estimates():
    case = "experiments_redone/impulse"
    n = 10
    sensor_data_impulse = np.loadtxt("../data/masters_data/processed_data/impulse_sensor.csv", delimiter=",")

    tet, det = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_impulse)
    tel1, del1 = get_states("estimates/" + case + "_experiment/l1_lam001_", n, case, sensor_data_impulse)
    tehp, dehp = get_states("estimates/" + case + "_experiment/hp_trend_lam01_", n, case, sensor_data_impulse)
    tel1t, del1t = get_states("estimates/" + case + "_experiment/l1_trend_lam10_", n, case, sensor_data_impulse)

    ax1 = plt.subplot(221)
    plt.title("a)", loc='left')
    ax1.plot(sensor_data_impulse[:,0], sensor_data_impulse[:,-1], color='black', label='Measurement')
    ax1.plot(tet, det, color='C3', linestyle='dashed', label='Tikhonov regularization')
    ax1.plot(tel1, del1, color='C0', linestyle='dashed', label='$\ell_1$ regularization', alpha=0.8)
    ax1.plot(tehp, dehp, color='red', label='H-P trend filtering')
    ax1.plot(tel1t, del1t, color='blue', label='$\ell_1$ trend filtering')
    ax1.set_xlim(1.1,1.6)
    ax1.set_ylim(2,21)
    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(1.1, 1.38),
        fancybox=True,
        shadow=True,
        ncol=3
    )
    ax1.set_ylabel("Torque (Nm)")
    ax1.set_xlabel("Time (s)")

    case = "experiments_redone/sin"
    n = 18
    sensor_data_sin = np.loadtxt("../data/masters_data/processed_data/sin_sensor.csv", delimiter=",")

    tets, dets = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_sin)
    tel1s, del1s = get_states("estimates/" + case + "_experiment/l1_lam01_", n, case, sensor_data_sin)
    tehps, dehps = get_states("estimates/" + case + "_experiment/hp_trend_lam10_", n, case, sensor_data_sin)
    tel1ts, del1ts = get_states("estimates/" + case + "_experiment/l1_trend_lam1_", n, case, sensor_data_sin)

    plt.subplot(222)
    plt.title("b)", loc='left')
    plt.plot(sensor_data_sin[:,0], sensor_data_sin[:,-1], color='black', label='Torque transducer 2')
    plt.plot(tets, dets, color='C3', linestyle='dashed', label='Tikhonov regularization')
    plt.plot(tel1s, del1s, color='C0', linestyle='dashed', label='$\ell_1$-regularization', alpha=0.8)
    plt.plot(tehps, dehps, color='red', label='H-P trend filtering')
    plt.plot(tel1ts, del1ts, color='blue', label='$\ell_1$ trend filtering')
    plt.xlim(1.49,1.55)
    plt.ylim(-1,31)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    case = "experiments_redone/step"
    n = 25
    sensor_data_step = np.loadtxt("../data/masters_data/processed_data/step_sensor.csv", delimiter=",")

    tetst, detst = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_step)
    tel1st, del1st = get_states("estimates/" + case + "_experiment/l1_lam1_", n, case, sensor_data_step)
    tehpst, dehpst = get_states("estimates/" + case + "_experiment/hp_trend_lam01_", n, case, sensor_data_step)
    tel1tst, del1tst = get_states("estimates/" + case + "_experiment/l1_trend_lam1_", n, case, sensor_data_step)

    plt.subplot(223)
    plt.title("c)", loc='left')
    plt.plot(sensor_data_step[:,0], sensor_data_step[:,-1], color='black', label='Torque transducer 2')
    plt.plot(tetst, detst, color='C3', linestyle='dashed', label='Tikhonov regularization')
    plt.plot(tel1st, del1st, color='C0', linestyle='dashed', label='$\ell_1$-regularization', alpha=0.8)
    plt.plot(tehpst, dehpst, color='red', label='H-P trend filtering')
    plt.plot(tel1tst, del1tst, color='blue', label='$\ell_1$ trend filtering')
    plt.xlim(4.2,5.3)
    plt.ylim(-1,27)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    case = "experiments_redone/ramp"
    n = 136
    sensor_data_ramp = np.loadtxt("../data/masters_data/processed_data/ramp_sensor.csv", delimiter=",")

    tetr, detr = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_ramp)
    tel1r, del1r = get_states("estimates/" + case + "_experiment/l1_lam001_", n, case, sensor_data_ramp)
    tehpr, dehpr = get_states("estimates/" + case + "_experiment/hp_trend_lam10_", n, case, sensor_data_ramp)
    tel1tr, del1tr = get_states("estimates/" + case + "_experiment/l1_trend_lam10_", n, case, sensor_data_ramp)

    plt.subplot(224)
    plt.title("d)", loc='left')
    plt.plot(sensor_data_ramp[:,0], sensor_data_ramp[:,-1], color='black', label='Torque transducer 2')
    plt.plot(tetr, detr, color='C3', linestyle='dashed', label='Tikhonov regularization')
    plt.plot(tel1r, del1r, color='C0', linestyle='dashed', label='$\ell_1$-regularization', alpha=0.8)
    plt.plot(tehpr, dehpr, color='red', label='H-P trend filtering')
    plt.plot(tel1tr, del1tr, color='blue', label='$\ell_1$ trend filtering')
    # plt.xlim(34.2,35.8)
    plt.xlim(34.5,35.5)
    plt.ylim(2,9)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    # plt.savefig("../figures/unit_torque_estimates.pdf")
    plt.show()


def plot_kf_vs_hp():
    case = "step"
    n = 25
    sensor_data_ramp = np.loadtxt("../data/masters_data/processed_data/" + case + "_sensor.csv", delimiter=",")

    t_hp, d_hp = get_states("estimates/akf_vs_hp/step_hp_lam01_", n, case, sensor_data_ramp)

    with open("estimates/akf_vs_hp/step_KF.pickle", 'rb') as handle_kf:
        kf_dataset = pickle.load(handle_kf)
        time_kf = kf_dataset[0]
        input_kf = kf_dataset[1]
        torque_kf = kf_dataset[2]

    with open("estimates/akf_vs_hp/step_0_meanKF.pickle", 'rb') as handle_kf:
        kf_dataset = pickle.load(handle_kf)
        time_kf_0 = kf_dataset[0]
        input_kf_0 = kf_dataset[1]
        torque_kf_0 = kf_dataset[2]


    plt.subplot(211)
    plt.title("a)", loc='left')
    plt.plot(sensor_data_ramp[:,0], sensor_data_ramp[:,-1], color='black', label='Measurement')
    plt.plot(t_hp, d_hp, color='red', label='H-P trend filtering')
    plt.plot(time_kf[:-2], torque_kf[2:,18], color='royalblue', label='AKF')
    plt.legend()
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")
    plt.xlim(4.1,5.2)
    plt.ylim(-2,27)

    plt.subplot(212)
    plt.title("b)", loc='left')
    plt.plot(sensor_data_ramp[:,0], sensor_data_ramp[:,-1], color='black', label='Measurement')
    plt.plot(t_hp, d_hp, color='red', label='H-P trend filtering')
    plt.plot(time_kf_0[:-2], torque_kf_0[2:,18], color='royalblue', label='AKF (zero mean motor torque)')
    plt.legend()
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")
    plt.xlim(4.1,5.2)
    plt.ylim(-2,27)

    plt.tight_layout()
    plt.savefig("../figures/step_kf_both_vs_hp.pdf")
    plt.show()


if __name__ == "__main__":
    # plot_unit_estimates()
    # plot_maritime_estimates()
    plot_kf_vs_hp()

    # plot_ice_different_lambda()
