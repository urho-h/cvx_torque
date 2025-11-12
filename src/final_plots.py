import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.linalg as LA
import scienceplots
import pickle
from scipy.signal import dlsim, butter, lfilter, welch
from joblib import Parallel, delayed

import opentorsion as ot

import data_equation as deq
import testbench_model_minimal as tbm
from full_scale import full_scale_model_minimal as fsm


np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '100'})


def impulse_excitation(sim_times, plot=False):
    """
    Parameters:

    sim_times : numpy.ndarray
        Timesteps of the simulation.

    Returns:

    U_imp : numpy.ndarray
        Input torque matrix. Rows correspond to timesteps, first column has the motor torque,
        second column has the propeller torque.
    """
    U_imp = np.zeros((len(sim_times), 2))

    # e1 = np.random.normal(0, .1, U_imp.shape[0])
    # e2 = np.random.normal(0, .1, U_imp.shape[0])

    one_hit = -np.hstack((np.linspace(0, 0.34, 7), np.linspace(0.38, 0.64, 7), np.linspace(0.67, 0.87, 7), np.linspace(0.9, 0.98, 7)))

    U_imp[:,0] += 2.7 # + e1
    # U_imp[:,1] += e2
    U_imp[3200:3228,1] += one_hit*10
    U_imp[3228:3228+28,1] += np.flip(one_hit*10)
    U_imp[5200:5228,1] += one_hit*10
    U_imp[5228:5228+28,1] += np.flip(one_hit*10)
    U_imp[8200:8228,1] += one_hit*10
    U_imp[8228:8228+28,1] += np.flip(one_hit*10)

    if plot:
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


def simulated_impulse_experiment(save_fig=False, show_plot=False):
    fs = 1000  # sampling frequency
    t_span = (0,10)  # simulation start and end times
    t_sim = np.arange(t_span[0], t_span[1], 1/fs)  # timesteps
    dt_sim = np.mean(np.diff(t_sim))  # timestep length
    impulse_sim = impulse_excitation(t_sim)

    # excitation with one impulse
    t_sim_short = t_sim[:1000]
    excitation_sim = impulse_sim[2500:3500]

    A, B, C, D, c, d = tbm.get_testbench_state_space(dt_sim)
    ss = (A, B, c, d, dt_sim)
    ss2 = (A, B, C, D, dt_sim)

    tout_sim, yout_sim, xout_sim = dlsim(ss2, excitation_sim, t=t_sim_short)  # simulate

    measurements = yout_sim + np.random.normal(0, .1, yout_sim.shape)  # add measurement noise

    # estimation parameters
    lam1 = 1e-1
    lam2 = 10
    batch_size = 500

    # call estimation function
    phi_sim, state_ests = deq.ell2_analytical(
        ss,
        ss2,
        measurements[:,:-1],
        batch_size,
        0,
        t_sim_short,
        lam1=lam1,
        lam2=lam2,
        use_trend_filter=True
    )

    motor_estimates_sim = np.concatenate((phi_sim[0][43::2], phi_sim[1][43::2]))
    propeller_estimates_sim = np.concatenate((phi_sim[0][44::2], phi_sim[1][44::2]))
    prop_shaft_estimates = np.concatenate((state_ests[0][3::4], state_ests[1][3::4]))

    if show_plot:
        # Create a figure
        fig = plt.figure(figsize=(6, 4))

        # Define the grid layout with 3 rows and 2 columns
        gs = gridspec.GridSpec(3, 2, height_ratios=[0.5, 0.5, 1])

        # Create the first four subplots (2x2 grid)
        ax1 = fig.add_subplot(gs[0, 0])  # Row 0, Column 0
        ax2 = fig.add_subplot(gs[0, 1])  # Row 0, Column 1
        ax3 = fig.add_subplot(gs[1, 0])  # Row 1, Column 0
        ax4 = fig.add_subplot(gs[1, 1])  # Row 1, Column 1

        # Create the bottom subplot that spans both columns
        ax5 = fig.add_subplot(gs[2, :])  # Row 2, spans all columns

        # Plot data
        ax1.plot(np.arange(1, 22, 1), xout_sim[500,:][:21], label=r'True $\tau(0)$')
        ax1.plot(np.arange(1, 22, 1), phi_sim[1][:21], 'b-', label=r'Estimated $\tau(0)$')
        ax1.set_xlabel("Node (-)", fontsize=11)
        ax1.set_ylabel("Torque (Nm)", fontsize=11)
        ax1.legend(loc='upper center', ncol=2)
        ax1.set_ylim(-2,6.5)
        ax1.minorticks_off()
        ax2.plot(np.arange(1, 23, 1), xout_sim[500,:][21:], label=r'True $\dot{\theta}(0)$')
        ax2.plot(np.arange(1, 23, 1), phi_sim[1][21:43], 'b-', label=r'Estimated $\dot{\theta}(0)$')
        ax2.set_xlabel("Node (-)", fontsize=11)
        ax2.set_ylabel("Speed (rad/s)", fontsize=11)
        ax2.legend()
        ax2.minorticks_off()

        ax3.plot(t_sim_short, excitation_sim[:,0], label=r'True $u_{\mathrm{m}}$')
        ax3.plot(t_sim_short, motor_estimates_sim, 'b-', label=r'Estimated $u_{\mathrm{m}}$')#'Simulated driving motor input estimate')
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Torque (Nm)", fontsize=11)
        ax3.set_ylim(2.0, 4.5)
        ax3.legend(ncol=2)
        ax3.minorticks_off()
        ax4.plot(t_sim_short, excitation_sim[:,1], label=r'True $u_{\mathrm{p}}$')
        ax4.plot(t_sim_short, propeller_estimates_sim, 'b-', label=r'Estimated $u_{\mathrm{p}}$')
        ax4.set_xlabel("Time (s)", fontsize=11)
        ax4.set_ylabel("Torque (Nm)", fontsize=11)
        ax4.legend()
        ax4.minorticks_off()

        ax5.plot(t_sim_short, yout_sim[:,-1], label=r"True propeller shaft torque $\tau_{20}$")
        ax5.plot(t_sim_short, prop_shaft_estimates, '-', c='b', label=r"Propeller shaft torque estimate $\tau_{20}$")
        ax5.set_xlabel("Time (s)", fontsize=11)
        ax5.set_ylabel("Torque (Nm)", fontsize=11)
        ax5.legend()
        ax5.minorticks_off()

        # Add some titles or labels for clarity
        ax1.set_title('a)', loc='left', fontsize=12)
        ax2.set_title('b)', loc='left', fontsize=12)
        ax3.set_title('c)', loc='left', fontsize=12)
        ax4.set_title('d)', loc='left', fontsize=12)
        ax5.set_title('e)', loc='left', fontsize=12)

        # Adjust the layout to prevent overlap
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # RMSE values
        tau0_rmse = np.sqrt(np.mean((xout_sim[500,:][:21] - phi_sim[1][:21].T)**2))
        theta0_rmse = np.sqrt(np.mean((xout_sim[500,:][21:] - phi_sim[1][21:43].T)**2))
        motor_rmse = np.sqrt(np.mean((excitation_sim[:,0] - motor_estimates_sim.T)**2))
        prop_rmse = np.sqrt(np.mean((excitation_sim[:,1] - propeller_estimates_sim.T)**2))
        prop_shaft_rmse = np.sqrt(np.mean((yout_sim[:,-1] - prop_shaft_estimates.T)**2))
        print("RMSE values:")
        print("tau0: ", tau0_rmse)
        print("theta0: ", theta0_rmse)
        print("motor: ", motor_rmse)
        print("propeller: ", prop_rmse)
        print("propeller shaft: ", prop_shaft_rmse)

        # Show the plot
        if save_fig:
            plt.savefig("figs/simulated_impulse_estimates.pdf")
        plt.show()

    return


def monte_carlo_impulse_simulation(n_sim=1000):
    fs = 1000  # sampling frequency
    t_span = (0, 10)  # simulation start and end times
    t_sim = np.arange(t_span[0], t_span[1], 1/fs)  # timesteps
    dt_sim = np.mean(np.diff(t_sim))  # timestep length
    impulse_sim = impulse_excitation(t_sim)

    # excitation with one impulse
    t_sim_short = t_sim[:1000]
    excitation_sim = impulse_sim[2500:3500]

    A, B, C, D, c, d = tbm.get_testbench_state_space(dt_sim)
    ss = (A, B, c, d, dt_sim)
    ss2 = (A, B, C, D, dt_sim)

    tout_sim, yout_sim, xout_sim = dlsim(ss2, excitation_sim, t=t_sim_short)  # simulate

    def estimation(t_sim_short, yout_sim, ss, ss2):
        measurements = yout_sim + np.random.normal(0, .1, yout_sim.shape)  # add measurement noise

        # estimation parameters
        lam1 = 1e-1
        lam2 = 10
        batch_size = 500

        # call estimation function
        phi_sim, state_ests = deq.ell2_analytical(
            ss,
            ss2,
            measurements[:,:-1],
            batch_size,
            0,
            t_sim_short,
            lam1=lam1,
            lam2=lam2,
            use_trend_filter=True
        )

        motor_estimates_sim = np.concatenate((phi_sim[0][43::2], phi_sim[1][43::2]))
        propeller_estimates_sim = np.concatenate((phi_sim[0][44::2], phi_sim[1][44::2]))
        prop_shaft_estimates = np.concatenate((state_ests[0][3::4], state_ests[1][3::4]))

        tau0_res = (xout_sim[500,:][:21] - phi_sim[1][:21].T)
        theta0_res = (xout_sim[500,:][21:] - phi_sim[1][21:43].T)
        motor_res = (excitation_sim[:,0] - motor_estimates_sim.T)
        prop_res = (excitation_sim[:,1] - propeller_estimates_sim.T)
        prop_shaft_res = (yout_sim[:,-1] - prop_shaft_estimates.T)

        return tau0_res, theta0_res, motor_res, prop_res, prop_shaft_res

    def single_simulation(i):
        tau0_res, theta0_res, motor_res, prop_res, prop_shaft_res = estimation(t_sim_short, yout_sim, ss, ss2)

        tau0_rmse = np.mean(tau0_res**2)
        theta0_rmse = np.mean(theta0_res**2)
        motor_rmse = np.mean(motor_res**2)
        prop_rmse = np.mean(prop_res**2)
        prop_shaft_rmse = np.mean(prop_shaft_res**2)

        return tau0_rmse, theta0_rmse, motor_rmse, prop_rmse, prop_shaft_rmse

    # Run simulations in parallel
    results = Parallel(n_jobs=-1)(delayed(single_simulation)(i) for i in range(n_sim))

    # Aggregate results
    tau0_rmse = np.sqrt(np.mean([res[0] for res in results]))
    theta0_rmse = np.sqrt(np.mean([res[1] for res in results]))
    motor_rmse = np.sqrt(np.mean([res[2] for res in results]))
    prop_rmse = np.sqrt(np.mean([res[3] for res in results]))
    prop_shaft_rmse = np.sqrt(np.mean([res[4] for res in results]))

    output_file = "simulation_results_2000.txt"
    with open(output_file, "w") as f:
        f.write("Monte Carlo Simulation Results\n")
        f.write("-------------------------------\n")
        f.write(f"tau0 RMSE: {tau0_rmse}\n")
        f.write(f"theta0 RMSE: {theta0_rmse}\n")
        f.write(f"motor RMSE: {motor_rmse}\n")
        f.write(f"prop RMSE: {prop_rmse}\n")
        f.write(f"prop_shaft RMSE: {prop_shaft_rmse}\n")

    print(f"Results written to {output_file}")
    print(tau0_rmse)
    print(theta0_rmse)
    print(motor_rmse)
    print(prop_rmse)
    print(prop_shaft_rmse)

    return tau0_rmse, theta0_rmse, motor_rmse, prop_rmse, prop_shaft_rmse


def process_estimates(n_batches, estimates, nstates=43):
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
            motor_estimates = all_motor_estimates[:-2]
            all_propeller_estimates = estimates[i][(nstates+1)::2]
            prop_est_overlap.append(all_propeller_estimates)
            propeller_estimates = all_propeller_estimates[:-2]
        else:
            all_motor_estimates = estimates[i][nstates::2]
            motor_est_overlap.append(all_motor_estimates)
            motor_estimates = np.concatenate(
                (motor_estimates, all_motor_estimates[1:-1])
            )
            all_propeller_estimates = estimates[i][(nstates+1)::2]
            prop_est_overlap.append(all_propeller_estimates)
            propeller_estimates = np.concatenate(
                (propeller_estimates, all_propeller_estimates[1:-1])
            )

    return motor_estimates, propeller_estimates, estimates[i][:44]


def ice_excitation_experiment(save_fig=False, show_plot=False):
    dt = 1e-3
    A, B, C, D, c, d = tbm.get_testbench_state_space(dt)
    ss = (A, B, c, d, dt)
    ss2 = (A, B, C, D, dt)

    # The motor dataset columns:
    # | time | motor setpoint | motor | propeller setpoint | propeller |
    motor_data = np.loadtxt("../data/masters_data/processed_data/ice_2000_motor.csv", delimiter=",")

    # The sensor dataset columns:
    # | time | speed1 (encoder1) | speed2 (encoder2) | torque1 | torque2 |
    sensor_data = np.loadtxt("../data/masters_data/processed_data/ice_2000_sensor.csv", delimiter=",")
    time = sensor_data[:,0]
    measurements = sensor_data[:,1:] # measurement data from the encoder 1, encoder 2, torque transducer 1 and torque transducer 2
    motors = np.vstack((motor_data[:,2], motor_data[:,4])).T # motor torque data

    motor_setpoint = motor_data[:,1]
    propeller_setpoint = motor_data[:,3]
    motor = motors[:,0]
    propeller = motors[:,-1]

    lam_imp1 = 1e-2
    lam_imp2 = 10
    batch_size = 500
    estimates_imp, state_imp = deq.ell2_analytical(
        ss,
        ss2,
        measurements[:,:3],
        batch_size,
        1,
        time,
        lam1=lam_imp1,
        lam2=lam_imp2,
        use_trend_filter=True
    )

    ice_state_estimates = np.concatenate(state_imp)
    t2_estimates = ice_state_estimates[3::4]
    motor_estimates_imp, propeller_estimates_imp, final_x = process_estimates(
        len(estimates_imp),
        estimates_imp
    )

    if show_plot:
        fig = plt.figure(figsize=(6, 4))
        gs = gridspec.GridSpec(2, 2, height_ratios=[0.5, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # Row 1, Column 1
        ax2 = fig.add_subplot(gs[0, 1])  # Row 1, Column 2
        ax3 = fig.add_subplot(gs[1, :])  # Row 2, spans all columns

        ax1.plot(motor_data[665:,0], motor_data[:-665,2], color='black', label=r'Measured $u_{\mathrm{m}}$')
        ax1.plot(time[:-(67)], motor_estimates_imp, 'b-', label=r'Estimated $u_{\mathrm{m}}$')
        ax1.set_xlabel("Time (s)", fontsize=11)
        ax1.set_ylabel("Torque (Nm)", fontsize=11)
        ax1.set_xlim(3.7,5.2)
        ax1.set_ylim(0,8)
        ax1.legend()
        ax1.minorticks_off()

        ax2.plot(motor_data[665:,0], motor_data[:-665,4], c='black', label=r'Measured $u_{\mathrm{p}}$')
        ax2.plot(time[:-67], -propeller_estimates_imp, 'b-', label=r'Estimated $u_{\mathrm{p}}$')
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.set_ylabel("Torque (Nm)", fontsize=11)
        ax2.set_xlim(3.7,5.2)
        ax2.set_ylim(-4,25)
        ax2.legend()
        ax2.minorticks_off()

        ax3.plot(time, measurements[:,-1], c='black', label=r'Measured $\tau_{20}$')
        ax3.plot(time[:-42], t2_estimates[15:], 'b-', label=r'Estimated $\tau_{20}$')
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Torque (Nm)", fontsize=11)
        ax3.set_xlim(3.7,5.2)
        ax3.legend()
        ax3.minorticks_off()

        # Add some titles or labels for clarity
        ax1.set_title('a)', loc='left', fontsize=12)
        ax2.set_title('b)', loc='left', fontsize=12)
        ax3.set_title('c)', loc='left', fontsize=12)

        # Adjust the layout to prevent overlap
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # plt.figure()
        # plt.plot(motor_data[3000:4400,2])
        # plt.plot(motor_estimates_imp[3700:5100])
        # plt.figure()
        # plt.plot(motor_data[3000:4400,4])
        # plt.plot(-propeller_estimates_imp[3690:5090])
        # plt.figure()
        # plt.plot(measurements[3500:5400,-1])
        # plt.plot(t2_estimates[3515:5415])
        # plt.show()

        # RMSE values
        motor_rmse = np.sqrt(np.mean((motor_data[3000:4400,2] - motor_estimates_imp[3700:5100])**2))
        prop_rmse = np.sqrt(np.mean((motor_data[3000:4400,4] - (-propeller_estimates_imp[3690:5090]))**2))
        prop_shaft_rmse = np.sqrt(np.mean((measurements[3500:5400,-1] - t2_estimates[3515:5415])**2))
        print("RMSE values:")
        print("motor: ", motor_rmse)
        print("propeller: ", prop_rmse)
        print("propeller shaft: ", prop_shaft_rmse)
        relative_error = abs(1 - np.divide(measurements[3500:5400,-1], t2_estimates[3515:5415].T))
        print(relative_error)
        plt.figure()
        plt.plot(relative_error[0])

        # Show the plot
        if save_fig:
            plt.savefig("figs/ice_experiment.pdf")
        plt.show()

    return


def full_scale_experiment(show_plot=False):
    dt = 1e-3
    A, B, C, D, c, d = fsm.get_full_scale_state_space(dt)
    ss = (A, B, c, d, dt)
    ss2 = (A, B, C, D, dt)

    # open pickled full scale data
    with open("full_scale/full_scale_data_unfiltered.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        time = dataset[0][:3000]
        motor_shaft_speed = dataset[1][:3000]  # rad/s
        motor_shaft_torque = dataset[2][:3000]  # Nm
        propeller_shaft_speed = dataset[3][:3000]  # rad/s
        propeller_shaft_torque = dataset[4][:3000]  # Nm

    measurements = np.stack((
        motor_shaft_speed,
        motor_shaft_torque,
        propeller_shaft_torque,
    )).T

    lam1 = 1e-4
    lam2 = 1
    batch_size = 1000
    estimates_imp, state_imp = deq.ell2_analytical(
        ss,
        ss2,
        measurements[:,:-1],
        batch_size,
        1,
        time,
        lam1=lam1,
        lam2=lam2,
        use_trend_filter=True,
        full_scale=True
    )

    ice_state_estimates = np.concatenate(state_imp[:-1])
    t2_estimates = ice_state_estimates[2::3]
    motor_estimates_imp, propeller_estimates_imp, final_x = process_estimates(
        len(estimates_imp[:-1]),
        estimates_imp[:-1],
        nstates=35
    )
    time = time[:-batch_size]
    fm, Pm = dft(time, measurements[:,-1])
    fy, Py = dft(time, t2_estimates.ravel())

    if show_plot:
        fig = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2]) #, height_ratios=[0.5, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # Row 1, Column 1
        ax2 = fig.add_subplot(gs[0, 1])  # Row 1, Column 2
        ax3 = fig.add_subplot(gs[1, :2])  # Row 2, spans all columns
        ax4 = fig.add_subplot(gs[:, 2])  # Row 2, spans all columns

        ax1.plot(time, motor_estimates_imp, 'b-', label=r'Estimated $u_{\mathrm{m}}$')
        ax1.set_xlabel("Time (s)", fontsize=11)
        ax1.set_ylabel("Torque (Nm)", fontsize=11)
        ax1.set_xlim(0,0.7)
        ax1.legend()
        ax1.minorticks_off()

        ax2.plot(time, -propeller_estimates_imp, 'b-', label=r'Estimated $u_{\mathrm{p}}$')
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.set_ylabel("Torque (Nm)", fontsize=11)
        ax2.set_xlim(0,0.7)
        ax2.legend()
        ax2.minorticks_off()

        ax3.plot(time, measurements[:-batch_size,-1], c='black', label=r'Measured $\tau_{16}$')
        ax3.plot(time, t2_estimates[:-4], 'b-', label=r'Estimated $\tau_{16}$')
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Torque (Nm)", fontsize=11)
        ax3.set_xlim(0,0.7)
        ax3.legend(ncol=2)
        ax3.minorticks_off()

        gr = (42/17)*(38/14)
        mot_mean_speed = np.mean(measurements[:,0]/(2*np.pi))
        prop_mean_speed = np.mean(measurements[:,0]/(2*np.pi*gr))
        # plt.plot(fgr, Pgr, label='naive estimate')
        ax4.plot(fm, Pm, 'r-', label='Measurement')
        ax4.plot(fy, Py, 'b-', label='Estimate', alpha=0.7)

        ax4.annotate(r'$\leftarrow \omega_m$', (mot_mean_speed, 500), (mot_mean_speed+0.3, 100))
        ax4.annotate(r'$\leftarrow \omega_p$', (prop_mean_speed, 500), (prop_mean_speed+0.3, 400))
        ax4.annotate(r'$\omega_n \rightarrow$', (prop_mean_speed*3, 500), (prop_mean_speed*3-3.5, 230))
        ax4.annotate(r'$\leftarrow \omega_p$ x 4', (prop_mean_speed*4, 500), (prop_mean_speed*4+0.3, 150))

        ax4.set_xlabel("Frequency (Hz)", fontsize=11)
        ax4.set_ylabel("Torque (Nm)", fontsize=11)
        ax4.set_xlim(-1,35)
        ax4.set_ylim(-1,500)
        ax4.legend()
        ax4.minorticks_off()

        # Add some titles or labels for clarity
        ax1.set_title('a)', loc='left', fontsize=12)
        ax2.set_title('b)', loc='left', fontsize=12)
        ax3.set_title('c)', loc='left', fontsize=12)
        ax4.set_title('d)', loc='left', fontsize=12)

        # Adjust the layout to prevent overlap
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Show the plot
        plt.savefig("figs/full_scale_estimation.pdf")
        plt.show()

    return


def dft(t, torque):

    def fft(time, torques):
        Fs = 1/np.mean(np.diff(time))  # Sampling frequency
        torque = torques - np.mean(torques)
        # L = len(torque)
        L = 2**14
        Y = np.fft.fft(torque, n=L)
        P2 = abs(Y/L)
        P1 = P2[0:int(L/2)]
        P1[1:-2] = 2*P1[1:-2]
        f = (Fs/L)*np.linspace(0, (L/2), int(L/2))

        return f, P1

    f, P = fft(t, torque)

    return f, P


def ice_experiment_motor_torque(save_fig=False, show_plot=False):
    dt = 1e-3
    A, B, C, D, c, d = tbm.get_testbench_state_space(dt)
    ss = (A, B, c, d, dt)
    ss2 = (A, B, C, D, dt)

    # The motor dataset columns:
    # | time | motor setpoint | motor | propeller setpoint | propeller |
    motor_data = np.loadtxt("../data/masters_data/processed_data/ice_2000_motor.csv", delimiter=",")

    # The sensor dataset columns:
    # | time | speed1 (encoder1) | speed2 (encoder2) | torque1 | torque2 |
    sensor_data = np.loadtxt("../data/masters_data/processed_data/ice_2000_sensor.csv", delimiter=",")
    time = sensor_data[:,0]
    measurements = sensor_data[:,1:] # measurement data from the encoder 1, encoder 2, torque transducer 1 and torque transducer 2
    motors = np.vstack((motor_data[:,2], motor_data[:,4])).T # motor torque data

    motor_setpoint = motor_data[:,1]
    propeller_setpoint = motor_data[:,3]
    motor = motors[:,0]
    propeller = motors[:,-1]

    measurements = np.stack((sensor_data[:,1], sensor_data[:,2], motor_data[:,2], sensor_data[:,4])).T

    lam_imp1 = 1e-2
    lam_imp2 = 10
    batch_size = 500
    estimates_imp, state_imp = deq.ell2_analytical(
        ss,
        ss2,
        measurements[:,:3],
        batch_size,
        1,
        time,
        lam1=lam_imp1,
        lam2=lam_imp2,
        use_trend_filter=True
    )

    ice_state_estimates = np.concatenate(state_imp)
    t2_estimates = ice_state_estimates[3::4]
    motor_estimates_imp, propeller_estimates_imp, final_x = process_estimates(
        len(estimates_imp),
        estimates_imp
    )

    if show_plot:
        fig = plt.figure(figsize=(6, 4))
        gs = gridspec.GridSpec(2, 2, height_ratios=[0.5, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # Row 1, Column 1
        ax2 = fig.add_subplot(gs[0, 1])  # Row 1, Column 2
        ax3 = fig.add_subplot(gs[1, :])  # Row 2, spans all columns

        ax1.plot(motor_data[665:,0], motor_data[:-665,2], color='black', label=r'Measured $u_{\mathrm{m}}$')
        ax1.plot(time[:-(67)], motor_estimates_imp, 'b-', label=r'Estimated $u_{\mathrm{m}}$')
        ax1.set_xlabel("Time (s)", fontsize=11)
        ax1.set_ylabel("Torque (Nm)", fontsize=11)
        ax1.set_xlim(3.7,5.2)
        ax1.set_ylim(0,8)
        ax1.legend()
        ax1.minorticks_off()

        ax2.plot(motor_data[665:,0], motor_data[:-665,4], c='black', label=r'Measured $u_{\mathrm{p}}$')
        ax2.plot(time[:-67], -propeller_estimates_imp, 'b-', label=r'Estimated $u_{\mathrm{p}}$')
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.set_ylabel("Torque (Nm)", fontsize=11)
        ax2.set_xlim(3.7,5.2)
        ax2.set_ylim(-4,25)
        ax2.legend()
        ax2.minorticks_off()

        ax3.plot(time, measurements[:,-1], c='black', label=r'Measured $\tau_{20}$')
        ax3.plot(time[:-42], t2_estimates[15:], 'b-', label=r'Estimated $\tau_{20}$')
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Torque (Nm)", fontsize=11)
        ax3.set_xlim(3.7,5.2)
        ax3.legend()
        ax3.minorticks_off()

        # Add some titles or labels for clarity
        ax1.set_title('a)', loc='left', fontsize=12)
        ax2.set_title('b)', loc='left', fontsize=12)
        ax3.set_title('c)', loc='left', fontsize=12)

        # Adjust the layout to prevent overlap
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # Show the plot
        if save_fig:
            plt.savefig("figs/ice_experiment.pdf")
        plt.show()

    return


if __name__ == "__main__":
    simulated_impulse_experiment(save_fig=True, show_plot=True)
    # _, _, _, _, _ = monte_carlo_impulse_simulation(n_sim=2000)
    # ice_excitation_experiment(save_fig=False, show_plot=True)
    # full_scale_experiment(show_plot=True)
    # ice_experiment_motor_torque(save_fig=False, show_plot=True)
