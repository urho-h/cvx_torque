import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_step_experiment(plot_kf_vs_tikh=False, plot_lasso_vs_tikh=False):
    fname = 'estimates/step_experiment_lam1_lam0001.pickle'

    with open(fname, 'rb') as handle:
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
    inputs_tikh = input_tikh[0]
    inputs_lasso = input_lasso[0]

    for i in range(1, len(input_tikh)):
        inputs_tikh = np.vstack((inputs_tikh, input_tikh[i]))
        inputs_lasso = np.vstack((inputs_lasso, input_lasso[i]))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 12,
        "figure.figsize": (6,4),
    })

    if plot_kf_vs_tikh:
        plt.figure()
        plt.subplot(311)
        plt.plot(times[:10000], motor[10000:], label='Measurement')
        plt.plot(times[:10000-10], input_estimates_kf[10:,0], label='Kalman filter')
        plt.plot(times[:10000], inputs_tikh[::2], label='H-P trend filter', alpha=0.8)
        plt.tick_params('x', labelbottom=False)

        plt.subplot(312)
        plt.plot(times[:10000], propeller[10000:], label='Measurement')
        plt.plot(times[:10000-10], input_estimates_kf[10:,1], label='Kalman filter')
        plt.plot(times[:10000], inputs_tikh[1::2], label='H-P trend filter', alpha=0.8)
        plt.ylabel('Torque (Nm)')
        plt.tick_params('x', labelbottom=False)

        plt.subplot(313)
        plt.plot(times[:10000], torques[:,-1], label='Measurement')
        plt.plot(times[:10000-10], torque_estimates_kf[10:,-1], label='Kalman filter')
        plt.plot(times[:10000], states_tikh[:,-1], label='Tikhonov regularization', alpha=0.8)
        plt.xlabel('Time (s)')

        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.48, 3.9),
            fancybox=True,
            shadow=False,
            ncol=3
        )

        # plt.savefig("step_experiment_kf_tikh_results.pdf")
        plt.show()

    if plot_lasso_vs_tikh:
        plt.figure()
        plt.subplot(311)
        plt.plot(times[:10000], motor[10000:], label='Measurement')
        plt.plot(times[:10000], inputs_tikh[::2], label='Tikhonov regularization', alpha=0.8)
        plt.plot(times[:10000], inputs_lasso[::2], label='$\ell_1$-regularization', alpha=0.8)
        plt.tick_params('x', labelbottom=False)
        # plt.xlim(5,5.5)

        plt.subplot(312)
        plt.plot(times[:10000], propeller[10000:], label='Measurement')
        plt.plot(times[:10000], inputs_tikh[1::2], label='Tikhonov regularization', alpha=0.8)
        plt.plot(times[:10000], inputs_lasso[1::2], label='$\ell_1$-regularization', alpha=0.8)
        plt.ylabel('Torque (Nm)')
        plt.tick_params('x', labelbottom=False)
        # plt.xlim(5,5.5)

        plt.subplot(313)
        plt.plot(times[:10000], torques[:,-1], label='Measurement')
        plt.plot(times[:10000], states_tikh[:,-1], label='Tikhonov', alpha=0.8)
        plt.plot(times[:10000], states_lasso[:,-1], label='$\ell_1$', alpha=0.8)
        plt.xlabel('Time (s)')
        plt.xlim(5,5.5)

        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 3.9),
            fancybox=True,
            shadow=False,
            ncol=3
        )

        plt.tight_layout()
        # plt.savefig("step_experiment_lasso_vs_tikh.pdf")

        plt.figure()
        plt.plot(times[:10000], propeller[10000:], label='Measurement')
        plt.plot(times[:10000], inputs_tikh[1::2], label='Tikhonov')
        plt.plot(times[:10000], inputs_lasso[1::2], label='$\ell_1$')
        plt.ylabel('Torque (Nm)')
        plt.xlabel('Time (s)')
        plt.xlim(5.1,5.22)
        plt.ylim(0,1.5)

        plt.legend()

        plt.tight_layout()
        # plt.savefig("step_experiment_lasso_zoom.pdf")
        plt.show()


if __name__ == "__main__":
    plot_step_experiment(plot_kf_vs_tikh=True, plot_lasso_vs_tikh=True)
