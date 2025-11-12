import sys

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy.linalg as LA
from scipy.signal import dlsim, butter, lfilter

import testbench_MSSP as tb

plt.style.use('science')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 11,
    "figure.figsize": (6,6),
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
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)
    C_mod = np.insert(C, C.shape[0], np.zeros((1, C.shape[1])), 0)
    C_mod[C.shape[0],22+18] += 2e4

    return A, B, C_mod, np.zeros((C_mod.shape[0], B.shape[1]))


def simulate(times, load, measurements):
    dt = np.mean(np.diff(times))
    A, B, C, D = get_testbench_state_space(dt)
    tout, yout, _ = dlsim((A, B, C, D, dt), u=load, t=times)

    print("Torque sensor 1:")
    print("simulation mean: ", np.mean(yout[5000:,-2]))
    print("measurement mean: ", np.mean(measurements[5000:,-2]), "\n")

    print("Torque sensor 2:")
    print("simulation mean: ", np.mean(yout[5000:,-1]))
    print("measurement mean: ", np.mean(measurements[5000:,-1]))

    plt.figure("torque1")
    plt.plot(tout, yout[:,-2], label="sim")
    plt.plot(measurements[:,0], measurements[:,-2], label="meas")
    plt.legend()

    plt.figure("torque2")
    plt.plot(tout, yout[:,-1], label="Torque 2 simulated")
    plt.plot(measurements[:,0], measurements[:,-1], label="Torque 2 measurement")
    plt.grid()
    plt.legend()
    plt.show()


def plot_unit_setpoint():
    impulse_data = np.loadtxt("../data/masters_data/raw_data/impulse_motor.csv", delimiter=",", skiprows=1)
    sin_data = np.loadtxt("../data/masters_data/raw_data/2000rpm_sin_148_motor.csv", delimiter=",", skiprows=1)
    ramp_data = np.loadtxt("../data/masters_data/raw_data/ramp_motor.csv", delimiter=",", skiprows=1)

    t_step = np.linspace(0, 2, 2000)
    U_step = np.zeros((2000, 2))
    U_step[:,0] += 2.7
    U_step[1000:,1] += 1.2

    ice_data = np.loadtxt("../data/masters_data/raw_data/ice_motor.csv", delimiter=",", skiprows=1)
    cfd_data = np.loadtxt("../data/masters_data/raw_data/CFD_motor.csv", delimiter=",", skiprows=1)

    plt.subplot(321)
    plt.title("a)", loc='left')
    plt.plot(impulse_data[:-79000,0], impulse_data[69000:-10000,5]*8, label="Loading\n motor\n setpoint", color='blue')
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    plt.subplot(322)
    plt.title("b)", loc='left')
    plt.plot(sin_data[:200,0], sin_data[100000:100200,5]*8, label="Loading\n motor\n setpoint", color='blue')
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    plt.subplot(323)
    plt.title("c)", loc='left')
    plt.plot(t_step, U_step[:,1]*8, label="Loading\n motor\n setpoint", color='blue')
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    plt.subplot(324)
    plt.title("d)", loc='left')
    plt.plot(ramp_data[:,0], ramp_data[:,3]*(2*np.pi/60), label="Loading\n motor\n setpoint", color='blue')
    plt.ylabel("Speed (rad/s)")
    plt.xlabel("Time (s)")

    plt.subplot(325)
    plt.title("e)", loc='left')
    plt.plot(ice_data[:1500,0], ice_data[58000:59500,5]*8, label="Loading\n motor\n setpoint", color='blue')
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    plt.subplot(326)
    plt.title("f)", loc='left')
    plt.plot(cfd_data[:40000,0], cfd_data[253500:293500,5]*8, label="Loading\n motor\n setpoint", color='blue')
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    # plt.savefig("../figures/all_setpoint.pdf")
    plt.show()


def simulation_and_process_data():
    """
    This function is for processing the master's thesis measurements. The measured inputs can be applied on the testbench model in a simulation and the result can be compared to measured output.
    """
    plot_sensor = True
    plot_motor = False
    check_sync = False
    run_simulation = False
    save_processed_data = False

    s, e = 0, -1 # full data, sensor data slicing, s=start e=end
    s, e = 260000, -180000 # no load 2000 rpm
    # s, e = 4500, -5000 # PRBS 2000 rpm
    # s, e = 42000, -7000 # impulse 2000 rpm, single impulse
    # s, e = 69000, -20000 # step 2000 rpm, single step
    # s, e = 300000, -370000 # sinusoidal load steady state 2000 rpm
    # s, e = 9000, -190000 # ramp dataset
    # s, e = 9000, -10000 # ice dataset
    # s, e = 900, -2000 # CFD dataset
    # s, e = 168000, -3000 # ice 2000 rpm dataset
    # s, e = 774000, -210000 # CFD 2000 rpm dataset
    start = int(s/3) # start point for motor data, 1/3 of sensor data start

    sensor_data = np.loadtxt(
        "../data/masters_data/raw_data/no_load_sensor.csv",
        delimiter=",",
        skiprows=1
    )

    motor_data = np.loadtxt(
        "../data/masters_data/raw_data/no_load_motor.csv",
        delimiter=",",
        skiprows=1
    )

    time_raw = sensor_data[s:e,0]*25e-9
    time = time_raw-time_raw[0]

    # if np.any(time < 0):
    #     time = np.linspace(
    #         0,
    #         len(sensor_data[s:e,0])/3012,
    #         len(sensor_data[s:e,0])
    #     )

    enc1_angle = (sensor_data[s:e,1])*(2*np.pi/20000)
    enc1_time_raw = sensor_data[s:e,2]*25e-9
    enc1_time = enc1_time_raw - enc1_time_raw[0]
    enc2_angle = (sensor_data[s:e,3])*(2*np.pi/20000)
    enc2_time_raw = sensor_data[s:e,4]*25e-9
    enc2_time = enc2_time_raw - enc2_time_raw[0]

    speed1 = np.gradient(enc1_angle, enc1_time)
    speed2 = np.gradient(enc2_angle, enc2_time)

    # DONE: torque sensor data acquisition got switched up
    torque2 = sensor_data[s:e,-2]*10 # torque1 in the dataset
    torque1 = sensor_data[s:e,-1]*4 # torque2 in the dataset

    # lowpass filter
    speed1 = low_pass_filter(speed1, 500, 3012)
    speed2 = low_pass_filter(speed2, 500, 3012)
    torque1 = low_pass_filter(torque1, 500, 3012)
    torque2 = low_pass_filter(torque2, 500, 3012)

    # cut off anomality in data, might be caused by the lowpass filter
    time = time[:-1800]
    speed1 = speed1[1800:]
    speed2 = speed2[1800:]
    torque1 = torque1[1800:]
    torque2 = torque2[1800:]

    if plot_sensor:
        plt.figure()
        plt.plot(time, speed1)
        plt.figure()
        plt.plot(time, speed2)
        plt.figure("torques")
        plt.plot(time, torque1)
        plt.plot(time, torque2)

        plt.show()

    measurements = np.vstack((time[::3], speed1[::3], speed2[::3], torque1[::3], torque2[::3])).T
    # measurements = np.vstack((time, speed1, speed2, torque1, torque2)).T

    start += 600
    time_motor = motor_data[:,0]
    time_motor = time_motor-time_motor[0]
    time_motor = time_motor[:len(time[::3])]
    motor_set = motor_data[start:len(time[::3])+start,1]
    motor = motor_data[start:len(time[::3])+start,2]
    propeller_set = motor_data[start:len(time[::3])+start,5]*8
    propeller = motor_data[start:len(time[::3])+start,6]*8

    if check_sync:
        plt.figure("Torque 1")
        plt.plot(time_motor, motor)
        plt.plot(time, torque1)
        plt.figure("Torque 2")
        plt.plot(time_motor, propeller)
        plt.plot(time, torque2)
        plt.show()

    if plot_motor:
        plt.figure()
        plt.plot(time_motor, motor_set)
        plt.figure()
        plt.plot(time_motor, motor)
        plt.figure()
        plt.plot(time_motor, propeller_set)
        plt.figure()
        plt.plot(time_motor, propeller)

        plt.show()

    motor_measurements = np.vstack(
        (time_motor,
         motor_set,
         motor,
         propeller_set,
         propeller)
    ).T

    if run_simulation:
        load = np.vstack((motor, propeller)).T
        simulate(time_motor, load, measurements)

    if save_processed_data:
        np.savetxt(
            "../data/masters_data/processed_data/no_load_2000_sensor.csv",
            measurements,
            delimiter=","
        )
        np.savetxt(
            "../data/masters_data/processed_data/no_load_2000_motor.csv",
            motor_measurements,
            delimiter=","
        )


def process_baseline_data(rot, case, pickle_data=False):
    sensor_data = np.loadtxt(
        (str(rot)).join(("../data/gear_loss/sensor/", case)),
        delimiter=",",
        skiprows=1
    )

    s, e = int(110*rot), -int(110*rot)
    # s, e = int(2e5), -int(2e5)

    time_raw = sensor_data[s:e,0]
    time = time_raw-time_raw[0]

    enc2_angle = (sensor_data[s:e,3])*(2*np.pi/360)
    enc2_time_raw = sensor_data[s:e,4]
    enc2_time = enc2_time_raw - enc2_time_raw[0]
    enc4_angle = (sensor_data[s:e,7])*(2*np.pi/360)
    enc4_time_raw = sensor_data[s:e,8]
    enc4_time = enc4_time_raw - enc4_time_raw[0]

    speed2 = np.gradient(enc2_angle, enc2_time)
    speed4 = np.gradient(enc4_angle, enc4_time)

    # torque sensors mixed up
    torque2 = sensor_data[s:e,-2]
    torque1 = sensor_data[s:e,-1]

    speed2 = low_pass_filter(speed2, 500, 3012)
    speed4 = low_pass_filter(speed4, 500, 3012)
    torque1 = low_pass_filter(torque1, 500, 3012)
    torque2 = low_pass_filter(torque2, 500, 3012)

    motor_shaft_power = torque1*speed2
    prop_shaft_power = torque2*speed4
    power_loss = motor_shaft_power-prop_shaft_power

    damping = (torque1-1/12*torque2)/(speed2+1/4*speed4)

    # if pickle_data:
    #     pickle_fn = (rot).join(("../data/gear_loss/pickle/", "rpm.pickle"))
    #     with open(fname + "KF.pickle", 'wb') as handle:
    #         pickle.dump(
    #             [times_kf, input_estimates_kf, torque_estimates_kf],
    #             handle,
    #             protocol=pickle.HIGHEST_PROTOCOL
    #         )

    return np.mean(damping)


if __name__ == "__main__":
    # simulation_and_process_data()
    # plot_unit_setpoint()

    speeds = [250, 500, 750, 1000, 1250, 1500]
    damping_means = []
    case = "rpm_CT_baseline_6%_GP1_0.csv"
    case = "rpm_No_torque%_GP1.csv"
    for rot in speeds:
        damping_means.append(process_baseline_data(rot, case, pickle_data=False))

    plt.plot(speeds, damping_means)
    plt.plot(speeds, np.array(damping_means)*np.array(speeds))
    plt.show()
