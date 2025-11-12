import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy.linalg as LA
from scipy.signal import dlsim, butter, lfilter
import pickle
import pandas as pd

import testbench_MSSP as tb
from convex_optimization import input_estimation as ie
from kalmanfilter import kalmanfilter as kf


plt.style.use('science')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 12,
    "figure.figsize": (7,6),
})


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def calculate_L_curve(times, dt, load, lambdas, use_l1=False, use_trend=False, show_plot=False, pickle_data=False, fname='l_curve_'):
    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    tout, yout, _ = dlsim((A, B, C, D, dt), u=load, t=times)
    e3 = np.random.normal(0, .1, yout.shape)
    y_noise = yout + e3

    l_norm, residual_norm = ie.L_curve(
        sys,
        y_noise,
        tout,
        lambdas,
        use_l1=use_l1,
        use_trend=use_trend
    )

    if show_plot:
        plt.yscale("log")
        plt.xscale("log")
        plt.scatter(residual_norm, l_norm, color='blue')
        plt.xlabel("$||y-\Gamma u||_2$")
        plt.ylabel("$||L u||_2$")
        plt.grid(which="both")
        plt.tight_layout()
        plt.show()

    if pickle_data:
        with open(fname + '.pickle', 'wb') as handle:
            pickle.dump(
                [l_norm, residual_norm, lambdas],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )


def calculate_pareto_curve(times, dt, load, lambdas, use_trend=False, pickle_data=False, fname='pareto_curve_'):
    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    tout, yout, _ = dlsim((A, B, C, D, dt), u=load, t=times)
    e3 = np.random.normal(0, .1, yout.shape)
    y_noise = yout + e3

    l1_norms, l2_norms, residual_norms = ie.pareto_curve(
        sys,
        load,
        y_noise,
        tout,
        lambdas,
        use_trend=use_trend
    )

    if pickle_data:
        with open(fname + '.pickle', 'wb') as handle:
            pickle.dump(
                [l1_norms, l2_norms, residual_norms, lambdas],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )


def plot_elastic_net_curve():
    with open("estimates/pareto_curves/impulse_experiment_elastic_net_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l1_norms = np.array(dataset[0])
        l2_norms = np.array(dataset[1])
        res_norms = np.array(dataset[2])

    plt.figure("l1 norm")
    plt.imshow(l1_norms, cmap='viridis')
    plt.colorbar()

    plt.figure("l2 norm")
    plt.imshow(l2_norms, cmap='viridis')
    plt.colorbar()

    plt.figure("residual norm")
    plt.imshow(res_norms, cmap='viridis')
    plt.colorbar()
    plt.show()


def ramp_excitation(sim_times, fs, bs, plot_input=False):
    U_ramp = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_ramp.shape[0])
    e2 = np.random.normal(0, .01, U_ramp.shape[0])

    U_ramp[1200:3200,0] += np.linspace(0, 2.7, 2000)
    U_ramp[3200:6200,0] += 2.7
    U_ramp[6200:8200,0] += np.flip(np.linspace(0, 2.7, 2000))
    U_ramp[:,0] += e1
    U_ramp[:,1] += e2

    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_ramp[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_ramp[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return U_ramp


def step_excitation(sim_times, fs, bs, plot_input=False):
    U_step = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_step.shape[0])
    e2 = np.random.normal(0, .01, U_step.shape[0])

    # U_step[:,0] += np.flip(np.linspace(-2.5, 2.5, 10000)) + e1
    U_step[:,0] += 2.7 + e1
    U_step[:,1] += e2
    U_step[3200:5200,1] += 10.0
    U_step[5200:8200,1] -= 4.0
    U_step[8200:,1] += 0.5

    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_step[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_step[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return U_step


def impulse_excitation(sim_times, fs, bs, plot_input=False):
    U_imp = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_imp.shape[0])
    e2 = np.random.normal(0, .01, U_imp.shape[0])

    one_hit = np.hstack((np.linspace(0, 0.34, 7), np.linspace(0.38, 0.64, 7), np.linspace(0.67, 0.87, 7), np.linspace(0.9, 0.98, 7)))

    U_imp[:,0] += 2.7 + e1
    U_imp[:,1] += e2
    U_imp[3200:3228,1] += one_hit*10
    U_imp[3228:3228+28,1] += np.flip(one_hit*10)
    U_imp[5200:5228,1] += one_hit*10
    U_imp[5228:5228+28,1] += np.flip(one_hit*10)
    U_imp[8200:8228,1] += one_hit*10
    U_imp[8228:8228+28,1] += np.flip(one_hit*10)

    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_imp[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_imp[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return U_imp


def sinusoidal_excitation(sim_times, fs, bs, plot_input=False):

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

    freqs = [2000/60*2*np.pi, 4*2000/60*2*np.pi, 8*2000/60*2*np.pi]
    amps = [10, 5, 1]
    phases = [0, 0, 0]
    offset = 10
    sine_signal = sum_sines(freqs, amps, phases, sim_times, offset)

    U_sin = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_sin.shape[0])
    e2 = np.random.normal(0, .01, U_sin.shape[0])

    U_sin[:,0] += 2.7 + e1
    U_sin[:,1] += sine_signal + e2

    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_sin[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_sin[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return U_sin


def ice_excitation_simulated(sim_times, fs, bs, plot_input=False):
    one_hit = np.hstack((np.linspace(0, 0.34, 7), np.linspace(0.38, 0.64, 7), np.linspace(0.67, 0.87, 7), np.linspace(0.9, 0.98, 7)))
    ice_hit = np.hstack((one_hit, np.flip(one_hit)))
    ice_interaction = np.hstack((np.zeros(246), 1.5*ice_hit, 3.7*ice_hit, 6.0*ice_hit, 8.5*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 10*ice_hit, 8.5*ice_hit, 6.0*ice_hit, 3.7*ice_hit, 1.5*ice_hit, np.zeros(257)))

    U_ice = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_ice.shape[0])
    e2 = np.random.normal(0, .01, U_ice.shape[0])

    U_ice[:,0] += 2.7 + e1
    U_ice[:,1] += e2
    U_ice[2000:2000+len(ice_interaction),1] += ice_interaction
    U_ice[4000:4000+len(ice_interaction),1] += ice_interaction
    U_ice[6000:6000+len(ice_interaction),1] += ice_interaction

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

        # plt.savefig("ice_setpoint.pdf")
        plt.show()

    return U_ice


def input_and_state_estimation(load, meas, sim_times, batch_size, lam_tikh, lam_lasso, overlap=50, run_tikh=False, run_lasso=False, run_elastic_net=False, run_kf=False, use_trend_filter=False, pickle_results=False, fname='estimation_results.pickle'):
    dt = np.mean(np.diff(sim_times))
    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    # include second torque transducer
    C_mod = np.insert(C, 3, np.zeros((1, C.shape[1])), 0)
    C_mod[3,22+18] += 2e4
    D_mod = np.zeros((C_mod.shape[0], B.shape[1]))

    tout, yout, _ = dlsim((A, B, C_mod, D_mod, dt), u=load, t=sim_times)
    e3 = np.random.normal(0, .01, yout.shape)
    y_noise = yout + e3
    meas = y_noise

    n = len(sim_times)
    motor = load[:,0]
    propeller = load[:,-1]

    if run_tikh:
        input_tikh = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size,
            overlap,
            sim_times,
            lam=lam_tikh,
            use_trend_filter=use_trend_filter,
            pickle_data=pickle_results,
            fn=fname
        )

    if run_lasso:
        input_lasso = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size,
            overlap,
            sim_times,
            lam=lam_lasso,
            use_zero_init=True,
            use_lasso=True,
            use_trend_filter=use_trend_filter,
            pickle_data=pickle_results,
            fn=fname
        )

    if run_elastic_net:
        input_net = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size+2*overlap,
            sim_times,
            lam=lam_tikh,
            lam2=lam_lasso,
            use_zero_init=True,
            use_lasso=False,
            use_elastic_net=True,
            use_trend_filter=use_trend_filter,
            pickle_data=pickle_results,
            fn=fname
        )

    if run_kf:
        times_kf, input_estimates_kf, torque_estimates_kf = kf.run_kalman_filter(
            sim_times,
            np.vstack((meas[:,0], meas[:,1])),
            meas[:,2],
            meas[:,2:],
            load[:,0],
            load[:,-1],
            np.mean(load[:,0])
        )


def get_unit_test_loads(plot_input=False):
    fs = 1000
    sim_times = np.arange(0, 10, 1/fs)
    dt = np.mean(np.diff(sim_times))
    batch_size = 500

    impulse_load = impulse_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    sinusoidal_load = sinusoidal_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    step_load = step_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    ramp_load = ramp_excitation(sim_times, fs, batch_size, plot_input=plot_input)

    return sim_times, impulse_load, sinusoidal_load, step_load, ramp_load


def plot_unit_test_loads():
    sim_times, impulse_load, sinusoidal_load, step_load, ramp_load = get_unit_test_loads()

    plt.subplot(221)
    plt.title("a)", loc='left')
    plt.plot(sim_times[:200], impulse_load[3125:3325,1], color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")

    plt.subplot(222)
    plt.title("b)", loc='left')
    plt.plot(sim_times[:200], sinusoidal_load[3000:3200,1], color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")

    plt.subplot(223)
    plt.title("c)", loc='left')
    plt.plot(sim_times[:2000], step_load[2200:4200,1], color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")

    plt.subplot(224)
    plt.title("d)", loc='left')
    plt.plot(sim_times[:int(len(sim_times)/2)], ramp_load[:int(len(sim_times)/2),0], color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")

    plt.tight_layout()
    # plt.savefig("../figures/l_curve_setpoints_redone.pdf")
    plt.show()


def plot_input_estimates(plot_all=False):
    fs = 1000
    sim_times = np.arange(0, 10, 1/fs)
    batch_size = 500
    impulse_load = impulse_excitation(sim_times, fs, batch_size)
    sinusoidal_load = sinusoidal_excitation(sim_times, fs, batch_size)
    step_load = step_excitation(sim_times, fs, batch_size)
    ramp_load = ramp_excitation(sim_times, fs, batch_size)

    def get_estimate(fn):
        overlap = 50
        n_batches = 20
        motor_estimates, propeller_estimates = [], []
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

        return propeller_estimates, motor_estimates

    impulse_tikh, _ = get_estimate('estimates/simulated/impulse_experiment_tikh_lam001_')
    impulse_l1, _ = get_estimate('estimates/simulated/impulse_experiment_l1_lam001_')
    impulse_hp, _ = get_estimate('estimates/simulated/impulse_experiment_hp_trend_lam01_')
    impulse_l1t, _ = get_estimate('estimates/simulated/impulse_experiment_l1_trend_lam10_')

    if plot_all:
        ax1 = plt.subplot(221)
        # plt.title("a)", loc='left')
        ax1.plot(sim_times, impulse_load[:,-1], color='black', label='Simulated')
        ax1.plot(sim_times, impulse_tikh, color='dimgray', label='Tikhonov regularization')
        ax1.plot(sim_times, impulse_l1, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
        ax1.plot(sim_times, impulse_hp, color='red', label='H-P trend filtering')
        ax1.plot(sim_times, impulse_l1t, color='blue', label='$\ell_1$ trend filtering')
        ax1.set_xlim(3.15,3.3)
        ax1.set_ylim(-3,18)
        ax1.legend(
            loc='upper center',
            bbox_to_anchor=(1.1, 1.3),
            fancybox=True,
            shadow=True,
            ncol=3
        )
        ax1.set_ylabel("Torque (Nm)")
        ax1.set_xlabel("Time (s)")

        sin_tikh, _ = get_estimate('estimates/simulated/sin_experiment_tikh_lam001_')
        sin_l1, _ = get_estimate('estimates/simulated/sin_experiment_l1_lam01_')
        # sin_hp, _ = get_estimate('estimates/simulated/sin_experiment_hp_trend_lam10_')
        # sin_l1t, _ = get_estimate('estimates/simulated/sin_experiment_l1_trend_lam1_')
        sin_hp, sin_hp_mot = get_estimate('estimates/simulated/sin_experiment_hp_trend_lam01_')
        sin_l1t, _ = get_estimate('estimates/simulated/sin_experiment_l1_trend_lam01_')

        plt.subplot(222)
        # plt.title("b)", loc='left')
        plt.plot(sim_times, sinusoidal_load[:,-1], color='black', label='Torque transducer 2')
        plt.plot(sim_times, sin_tikh, color='dimgray', label='Tikhonov regularization')
        plt.plot(sim_times, sin_l1, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
        plt.plot(sim_times, sin_hp, color='red', label='H-P trend filtering')
        plt.plot(sim_times, sin_l1t, color='blue', label='$\ell_1$ trend filtering')
        plt.xlim(1.49,1.55)
        plt.ylim(-5,26)
        plt.ylabel("Torque (Nm)")
        plt.xlabel("Time (s)")

        step_tikh, _ = get_estimate('estimates/simulated/step_experiment_tikh_lam001_')
        step_l1, _ = get_estimate('estimates/simulated/step_experiment_l1_lam1_')
        step_hp, _ = get_estimate('estimates/simulated/step_experiment_hp_trend_lam01_')
        step_l1t, _ = get_estimate('estimates/simulated/step_experiment_l1_trend_lam1_')

        plt.subplot(223)
        # plt.title("c)", loc='left')
        plt.plot(sim_times, step_load[:,-1], color='black', label='Torque transducer 2')
        plt.plot(sim_times, step_tikh, color='dimgray', label='Tikhonov regularization')
        plt.plot(sim_times, step_l1, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
        plt.plot(sim_times, step_hp, color='red', label='H-P trend filtering')
        plt.plot(sim_times, step_l1t, color='blue', label='$\ell_1$ trend filtering')
        plt.xlim(3.0,3.4)
        plt.ylim(-1,15)
        plt.ylabel("Torque (Nm)")
        plt.xlabel("Time (s)")

        _, ramp_tikh = get_estimate('estimates/simulated/ramp_experiment_tikh_lam001_')
        _, ramp_l1 = get_estimate('estimates/simulated/ramp_experiment_l1_lam001_')
        _, ramp_hp = get_estimate('estimates/simulated/ramp_experiment_hp_trend_lam10_')
        _, ramp_l1t = get_estimate('estimates/simulated/ramp_experiment_l1_trend_lam10_')

        plt.subplot(224)
        # plt.title("d)", loc='left')
        plt.plot(sim_times, ramp_load[:,0], color='black', label='Torque transducer 2')
        plt.plot(sim_times, ramp_tikh, color='dimgray', label='Tikhonov regularization')
        plt.plot(sim_times, ramp_l1, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
        plt.plot(sim_times, ramp_hp, color='red', label='H-P trend filtering')
        plt.plot(sim_times, ramp_l1t, color='blue', label='$\ell_1$ trend filtering')
        plt.xlim(3.1,3.3)
        plt.ylim(2.4,2.8)
        plt.ylabel("Torque (Nm)")
        plt.xlabel("Time (s)")

        # plt.savefig("../figures/unit_torque_estimates.pdf")
        plt.show()
    else:
        plt.subplot(221)
        plt.title("Tikhonov regularization")
        plt.plot(sim_times, impulse_load[:,-1], color='black', label='Simulated input torque')
        plt.plot(sim_times, impulse_tikh, color='red', label='Estimated input torque')
        plt.xlim(3.12,3.33)
        plt.ylim(-3,16)
        plt.ylabel("Torque (Nm)")
        plt.xlabel("Time (s)")
        plt.legend(loc='upper left')
        plt.text(3.27, 6, "$\lambda = 0.01$")

        plt.subplot(222)
        plt.title("$\ell_1$ regularization")
        plt.plot(sim_times, impulse_load[:,-1], color='black', label='Simulated')
        plt.plot(sim_times, impulse_l1, color='red', linestyle='solid', label='$\ell_1$-regularization')
        plt.xlim(3.12,3.33)
        plt.ylim(-3,16)
        plt.ylabel("Torque (Nm)")
        plt.xlabel("Time (s)")
        plt.text(3.27, 6, "$\lambda = 0.01$")

        plt.subplot(223)
        plt.title("H-P trend filter")
        plt.plot(sim_times, impulse_load[:,-1], color='black', label='Simulated')
        plt.plot(sim_times, impulse_hp, color='red', label='H-P trend filtering')
        plt.xlim(3.12,3.33)
        plt.ylim(-3,16)
        plt.ylabel("Torque (Nm)")
        plt.xlabel("Time (s)")
        plt.text(3.27, 6, "$\lambda = 0.1$")

        plt.subplot(224)
        plt.title("$\ell_1$ trend filter")
        plt.plot(sim_times, impulse_load[:,-1], color='black', label='Torque transducer 2')
        plt.plot(sim_times, impulse_l1t, color='red', label='$\ell_1$ trend filtering')
        plt.xlim(3.12,3.33)
        plt.ylim(-3,16)
        plt.ylabel("Torque (Nm)")
        plt.xlabel("Time (s)")
        plt.text(3.27, 6, "$\lambda = 10$")

        plt.tight_layout()
        plt.savefig("../figures/unit_impulse_estimates.pdf")
        plt.show()


def simulation_experiment(run_estimation=False, plot_l_curve=False, plot_elastic_curve=False, plot_input=False):
    fs = 1000
    sim_times = np.arange(0, 10, 1/fs)
    dt = np.mean(np.diff(sim_times))
    batch_size = 500

    ramp_load = ramp_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    step_load = step_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    impulse_load = impulse_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    sinusoidal_load = sinusoidal_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    ice_load = ice_excitation_simulated(sim_times, fs, batch_size, plot_input=plot_input)

    load = sinusoidal_load

    if plot_input:
        A, B, C, D = get_testbench_state_space(dt)
        sys = (A, B, C, D)
        y_data_eq = ie.data_eq_simulation(sys, sim_times, load[:batch_size], batch_size)
        plt.plot(sim_times[:batch_size], y_data_eq[1::3], color='blue')
        plt.ylabel('Torque (Nm)')
        plt.xlabel('Time (s)')
        plt.show()

    if plot_l_curve:
        # s = 2400 # ramp
        s = 3000 # rest
        plt.plot(sim_times[:batch_size], load[s:s+batch_size, 0], label='Motor load', color='blue')
        plt.plot(sim_times[:batch_size], load[s:s+batch_size, 1], label='Propeller load', color='red')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.tight_layout()
        plt.show()

        calculate_L_curve(
            sim_times[:batch_size],
            dt,
            load[s:s+batch_size],
            lambdas=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000],
            use_l1=False,
            use_trend=True,
            show_plot=True,
            pickle_data=True,
            fname='estimates/pareto_curves_new/sin_experiment_hp_trend_curve'
            # fname='estimates/pareto_curves_new/sin_experiment_l1_trend_curve'
            # fname='estimates/pareto_curves_new/sin_experiment_l1_curve'
            # fname='estimates/pareto_curves_new/sin_experiment_tikh_curve'
        )

    if plot_elastic_curve:
        s = 3000
        calculate_pareto_curve(
            sim_times[:batch_size],
            dt,
            load[s:s+batch_size],
            lambdas=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
            use_trend=False,
            pickle_data=True,
            fname='estimates/pareto_curves/impulse_experiment_elastic_net_curve'
        )

    if run_estimation:
        input_and_state_estimation(
            load,
            0,
            sim_times,
            500,
            0.1,
            0.1,
            run_tikh=True,
            run_lasso=False,
            use_trend_filter=True,
            pickle_results=True,
            fname='estimates/simulated/sin_experiment_hp_trend_lam01_'
        )


if __name__ == "__main__":
    # simulation_experiment(
    #     plot_input=False,
    #     plot_l_curve=False,
    #     plot_elastic_curve=False,
    #     run_estimation=True
    # )

    # plot_elastic_net_curve()
    # plot_unit_test_loads()
    plot_input_estimates(plot_all=False)
