import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle

import simulation_pareto_curve as spc


plt.style.use('science')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 11,
    "figure.figsize": (10,5),
})


def plot_l_curves():
    sim_times, impulse_load, sinusoidal_load, step_load, ramp_load = spc.get_unit_test_loads()

    with open("estimates/pareto_curves_new/impulse_experiment_tikh_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_tikh = dataset[0]
        residual_norm_tikh = dataset[1]
        lambdas_tikh = dataset[2]

    with open("estimates/pareto_curves_new/impulse_experiment_l1_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1 = dataset[0]
        residual_norm_l1 = dataset[1]
        lambdas_l1 = dataset[2]

    with open("estimates/pareto_curves_new/impulse_experiment_hp_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_hp_trend = dataset[0]
        residual_norm_hp_trend = dataset[1]
        lambdas_hp_trend = dataset[2]

    with open("estimates/pareto_curves_new/impulse_experiment_l1_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1_trend = dataset[0]
        residual_norm_l1_trend = dataset[1]
        lambdas_l1_trend = dataset[2]

    plt.subplot(4, 5, 1)
    plt.title("Unit excitation")
    plt.plot(sim_times[:200], impulse_load[3125:3325,1], color='blue')
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])

    plt.subplot(4, 5, 2)
    plt.title("Tikhonov\nregularization")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_tikh[:-2], l_norm_tikh[:-2], color='blue')
    plt.scatter(residual_norm_tikh[5], l_norm_tikh[5], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_tikh[5]),
        (residual_norm_tikh[5], l_norm_tikh[5]),
        (residual_norm_tikh[5]+1, l_norm_tikh[5]+50)
    )
    plt.ylabel("$||L u||_2$")

    plt.subplot(4, 5, 3)
    plt.title("H-P trend filter")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_hp_trend, l_norm_hp_trend, color='blue')
    plt.scatter(residual_norm_hp_trend[6], l_norm_hp_trend[6], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[6]),
        (residual_norm_hp_trend[6], l_norm_hp_trend[6]),
        (residual_norm_hp_trend[6]+1, l_norm_hp_trend[6]+1)
    )
    plt.ylabel("$||L u||_2$")
    plt.xticks(ticks=[3,4,6,10], labels=[None,None,None,"$10^1$"])

    plt.subplot(4, 5, 4)
    plt.title("$\ell_1$-regularization")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1, l_norm_l1, color='blue')
    plt.scatter(residual_norm_l1[4], l_norm_l1[4], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1[5]),
        (residual_norm_l1[5], l_norm_l1[5]),
        (residual_norm_l1[5]+0.5, l_norm_l1[5]-1450)
    )
    plt.ylabel("$||L u||_1$")

    plt.subplot(4, 5, 5)
    plt.title("$\ell_1$ trend filter")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1_trend, l_norm_l1_trend, color='blue')
    plt.scatter(residual_norm_l1_trend[8], l_norm_l1_trend[8], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1_trend[8]),
        (residual_norm_l1_trend[8], l_norm_l1_trend[8]),
        (residual_norm_l1_trend[8]+0.5, l_norm_l1_trend[8]+100)
    )
    plt.ylabel("$||L u||_1$")
    plt.xticks(ticks=[3,4,6,10], labels=[None,None,None,"$10^1$"])
    # plt.ylim(top=11)

    with open("estimates/pareto_curves_new/sin_experiment_tikh_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_tikh = dataset[0]
        residual_norm_tikh = dataset[1]
        lambdas_tikh = dataset[2]

    with open("estimates/pareto_curves_new/sin_experiment_l1_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1 = dataset[0]
        residual_norm_l1 = dataset[1]
        lambdas_l1 = dataset[2]

    with open("estimates/pareto_curves_new/sin_experiment_hp_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_hp_trend = dataset[0]
        residual_norm_hp_trend = dataset[1]
        lambdas_hp_trend = dataset[2]

    with open("estimates/pareto_curves_new/sin_experiment_l1_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1_trend = dataset[0]
        residual_norm_l1_trend = dataset[1]
        lambdas_l1_trend = dataset[2]

    plt.subplot(4, 5, 6)
    plt.plot(sim_times[:200], sinusoidal_load[3000:3200,1], color='blue')
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])

    plt.subplot(4, 5, 7)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_tikh[:-2], l_norm_tikh[:-2], color='blue')
    plt.scatter(residual_norm_tikh[5], l_norm_tikh[5], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_tikh[5]),
        (residual_norm_tikh[5], l_norm_tikh[5]),
        (residual_norm_tikh[5]+5, l_norm_tikh[5]+2)
    )
    plt.ylabel("$||L u||_2$")

    plt.subplot(4, 5, 8)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_hp_trend, l_norm_hp_trend, color='blue')
    plt.scatter(residual_norm_hp_trend[8], l_norm_hp_trend[8], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[8]),
        (residual_norm_hp_trend[8], l_norm_hp_trend[8]),
        (residual_norm_hp_trend[8]+1, l_norm_hp_trend[8]+1)
    )
    plt.ylabel("$||L u||_2$")
    plt.xticks(ticks=[3,4,6,10], labels=[None,None,None,"$10^1$"])

    plt.subplot(4, 5, 9)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1, l_norm_l1, color='blue')
    plt.scatter(residual_norm_l1[6], l_norm_l1[6], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1[6]),
        (residual_norm_l1[6], l_norm_l1[6]),
        (residual_norm_l1[6]+1, l_norm_l1[6]+1100)
    )
    plt.ylabel("$||L u||_1$")

    plt.subplot(4, 5, 10)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1_trend, l_norm_l1_trend, color='blue')
    plt.scatter(residual_norm_l1_trend[7], l_norm_l1_trend[7], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1_trend[7]),
        (residual_norm_l1_trend[7], l_norm_l1_trend[7]),
        (residual_norm_l1_trend[7]+0.7, l_norm_l1_trend[7]+1)
    )
    plt.ylabel("$||L u||_1$")
    plt.xticks(ticks=[3,4,6,10], labels=[None,None,None,"$10^1$"])

    with open("estimates/pareto_curves_new/step_experiment_tikh_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_tikh = dataset[0]
        residual_norm_tikh = dataset[1]
        lambdas_tikh = dataset[2]

    with open("estimates/pareto_curves_new/step_experiment_l1_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1 = dataset[0]
        residual_norm_l1 = dataset[1]
        lambdas_l1 = dataset[2]

    with open("estimates/pareto_curves_new/step_experiment_hp_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_hp_trend = dataset[0]
        residual_norm_hp_trend = dataset[1]
        lambdas_hp_trend = dataset[2]

    with open("estimates/pareto_curves_new/step_experiment_l1_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1_trend = dataset[0]
        residual_norm_l1_trend = dataset[1]
        lambdas_l1_trend = dataset[2]

    plt.subplot(4, 5, 11)
    plt.plot(sim_times[:2000], step_load[2200:4200,1], color='blue')
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])

    plt.subplot(4, 5, 12)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_tikh[:-2], l_norm_tikh[:-2], color='blue')
    plt.scatter(residual_norm_tikh[5], l_norm_tikh[5], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_tikh[5]),
        (residual_norm_tikh[5], l_norm_tikh[5]),
        (residual_norm_tikh[5]+5, l_norm_tikh[5]+2)
    )
    plt.ylabel("$||L u||_2$")

    plt.subplot(4, 5, 13)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_hp_trend, l_norm_hp_trend, color='blue')
    plt.scatter(residual_norm_hp_trend[6], l_norm_hp_trend[6], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[6]),
        (residual_norm_hp_trend[6], l_norm_hp_trend[6]),
        (residual_norm_hp_trend[6]+1, l_norm_hp_trend[6]+10)
    )
    plt.ylabel("$||L u||_2$")

    plt.subplot(4, 5, 14)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1, l_norm_l1, color='blue')
    plt.scatter(residual_norm_l1[7], l_norm_l1[7], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1[7]),
        (residual_norm_l1[7], l_norm_l1[7]),
        (residual_norm_l1[7], l_norm_l1[7]-1000)
    )
    plt.ylabel("$||L u||_1$")

    plt.subplot(4, 5, 15)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1_trend, l_norm_l1_trend, color='blue')
    plt.scatter(residual_norm_l1_trend[8], l_norm_l1_trend[8], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1_trend[7]),
        (residual_norm_l1_trend[7], l_norm_l1_trend[7]),
        (residual_norm_l1_trend[7]+1, l_norm_l1_trend[7]+50)
    )
    plt.xticks(ticks=[3,4,6,10], labels=[None,None,None,"$10^1$"])
    plt.ylabel("$||L u||_1$")

    with open("estimates/pareto_curves_new/ramp_experiment_tikh_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_tikh = dataset[0]
        residual_norm_tikh = dataset[1]
        lambdas_tikh = dataset[2]

    with open("estimates/pareto_curves_new/ramp_experiment_l1_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1 = dataset[0]
        residual_norm_l1 = dataset[1]
        lambdas_l1 = dataset[2]

    with open("estimates/pareto_curves_new/ramp_experiment_hp_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_hp_trend = dataset[0]
        residual_norm_hp_trend = dataset[1]
        lambdas_hp_trend = dataset[2]

    with open("estimates/pareto_curves_new/ramp_experiment_l1_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1_trend = dataset[0]
        residual_norm_l1_trend = dataset[1]
        lambdas_l1_trend = dataset[2]

    plt.subplot(4, 5, 16)
    plt.plot(sim_times[:int(len(sim_times)/2)], ramp_load[:int(len(sim_times)/2),0], color='blue')
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])

    plt.subplot(4, 5, 17)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_tikh[:-2], l_norm_tikh[:-2], color='blue')
    plt.scatter(residual_norm_tikh[5], l_norm_tikh[5], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_tikh[5]),
        (residual_norm_tikh[5], l_norm_tikh[5]),
        (residual_norm_tikh[5]+2, l_norm_tikh[5]+80)
    )
    plt.xlabel("$||y-\Gamma u||_2$")
    plt.ylabel("$||L u||_2$")

    plt.subplot(4, 5, 18)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_hp_trend, l_norm_hp_trend, color='blue')
    plt.scatter(residual_norm_hp_trend[8], l_norm_hp_trend[8], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[8]),
        (residual_norm_hp_trend[8], l_norm_hp_trend[8]),
        (residual_norm_hp_trend[8]+1, l_norm_hp_trend[8])
    )
    plt.xticks(ticks=[3,4,6,10], labels=[None,None,None,"$10^1$"])
    plt.xlabel("$||y-\Gamma u||_2$")
    plt.ylabel("$||L u||_2$")

    plt.subplot(4, 5, 19)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1, l_norm_l1, color='blue')
    plt.scatter(residual_norm_l1[5], l_norm_l1[5], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1[5]),
        (residual_norm_l1[5], l_norm_l1[5]),
        (residual_norm_l1[5], l_norm_l1[5]-1000)
    )
    plt.xlabel("$||y-\Gamma u||_2$")
    plt.ylabel("$||L u||_1$")

    plt.subplot(4, 5, 20)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1_trend, l_norm_l1_trend, color='blue')
    plt.scatter(residual_norm_l1_trend[8], l_norm_l1_trend[8], color='red')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1_trend[8]),
        (residual_norm_l1_trend[8], l_norm_l1_trend[8]),
        (residual_norm_l1_trend[8]+1, l_norm_l1_trend[8])
    )
    plt.xticks(ticks=[3,4,6,10], labels=[None,None,None,"$10^1$"])
    plt.xlabel("$||y-\Gamma u||_2$")
    plt.ylabel("$||L u||_1$")

    plt.tight_layout()
    # plt.savefig("../figures/l_curves_all_redone.pdf")
    plt.show()

if __name__ == "__main__":
    plot_l_curves()
