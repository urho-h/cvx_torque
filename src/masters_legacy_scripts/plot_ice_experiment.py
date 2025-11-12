import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_ice_experiment():
    fname1 = 'estimates/ice_experiment_simulated_lam005.pickle'
    fname2 = 'estimates/ice_experiment_MSSP_measurements_lam005.pickle'

    with open(fname1, 'rb') as handle:
        dataset = pickle.load(handle)

        times_sim = dataset[0]
        motor_sim = dataset[1]
        propeller_sim = dataset[2]
        input_tikh_sim = dataset[3]
        input_lasso_sim = dataset[4]
        torques_sim = dataset[5] #TODO: check for speed measurements
        states_tikh_sim = dataset[6]
        states_lasso_sim = dataset[7]
        input_estimates_kf_sim = dataset[8]
        torque_estimates_kf_sim =  dataset[9]

    with open(fname2, 'rb') as handle:
        dataset = pickle.load(handle)

        times = dataset[0]
        motor = dataset[1]
        propeller = dataset[2]
        input_tikh = dataset[3]
        input_lasso = dataset[4]
        torques = dataset[5]
        states_tikh = dataset[6]
        states_lasso = dataset[7]
        input_estimates_kf = dataset[8]
        torque_estimates_kf =  dataset[9]

    # reshaping the estimates
    input_tikh_sim = np.vstack((input_tikh_sim[0], input_tikh_sim[1], input_tikh_sim[2]))
    input_lasso_sim = np.vstack((input_lasso_sim[0], input_lasso_sim[1], input_lasso_sim[2]))
    input_tikh = np.vstack((input_tikh[0], input_tikh[1], input_tikh[2]))
    input_lasso = np.vstack((input_lasso[0], input_lasso[1], input_lasso[2]))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 12,
    })

    plt.figure()
    plt.subplot(311)
    plt.plot(times, motor, label='Measurement')
    plt.plot(times[:-10], input_estimates_kf[10:,0], label='Kalman filter')
    plt.plot(times[:-1], input_tikh[::2], label='Tikhonov regularization', alpha=0.8)
    plt.ylim(-4,10)
    plt.tick_params('x', labelbottom=False)

    plt.subplot(312)
    plt.plot(times, propeller, label='Measurement')
    plt.plot(times[:-10], input_estimates_kf[10:,1], label='Kalman filter')
    plt.plot(times[:-1], input_tikh[1::2], label='Tikhonov regularization', alpha=0.8)
    plt.ylim(-5,20)
    plt.ylabel('Torque (Nm)')
    plt.tick_params('x', labelbottom=False)

    plt.subplot(313)
    plt.plot(times, torques[:,-1], label='Measurement')
    plt.plot(times[:-10], torque_estimates_kf[10:,-1], label='Kalman filter')
    plt.plot(times[:-2], states_tikh[:,-1], label='Tikhonov regularization', alpha=0.8)
    plt.ylim(0,26)
    plt.xlabel('Time (s)')

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 3.9),
        fancybox=True,
        shadow=False,
        ncol=3
    )

    plt.savefig("ice_experiment_kf_tikh_results.pdf")

    ##############################
    plt.figure()
    plt.subplot(321)
    plt.plot(times_sim, motor_sim, label='Measurement', color='blue')
    plt.plot(times_sim[:-1], input_lasso_sim[::2], label='$\ell_1$-regularization', alpha=0.8, color='red')
    plt.ylim(2.6,2.8)
    plt.tick_params('x', labelbottom=False)

    plt.subplot(322)
    plt.plot(times, motor, label='Measurement', color='blue')
    plt.plot(times[:-1], input_lasso[::2], label='$\ell_1$-regularization', alpha=0.8, color='red')
    plt.ylim(-3,9)
    plt.tick_params('x', labelbottom=False)

    plt.subplot(323)
    plt.plot(times_sim, propeller_sim, label='Measurement', color='blue')
    plt.plot(times_sim[:-1], input_lasso_sim[1::2], label='$\ell_1$-regularization', alpha=0.8, color='red')
    plt.ylim(-3,20)
    plt.ylabel('Torque (Nm)')
    plt.tick_params('x', labelbottom=False)

    plt.subplot(324)
    plt.plot(times, propeller, label='Measurement', color='blue')
    plt.plot(times[:-1], input_lasso[1::2], label='$\ell_1$-regularization', alpha=0.8, color='red')
    plt.ylim(-3,20)
    plt.tick_params('x', labelbottom=False)

    plt.subplot(325)
    plt.plot(times_sim, torques_sim[:,-1], label='Measurement', color='blue')
    plt.plot(times_sim[:-1], states_lasso_sim[:,-1], label='$\ell_1$-regularization', alpha=0.8, color='red')
    plt.ylim(-3,25)
    plt.xlabel('Time (s)')

    plt.subplot(326)
    plt.plot(times, torques[:,-1], label='Measurement', color='blue')
    plt.plot(times[:-2], states_lasso[:,-1], label='$\ell_1$-regularization', alpha=0.8, color='red')
    plt.ylim(-3,25)
    plt.xlabel('Time (s)')

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(-0.1, 3.9),
        fancybox=True,
        shadow=False,
        ncol=2
    )

    plt.savefig("ice_experiment_lasso_results.pdf")
    plt.show()


if __name__ == "__main__":
    plot_ice_experiment()
