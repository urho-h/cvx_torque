import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy.linalg as LA
from scipy.signal import dlsim, butter, lfilter
import pickle

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


def process_gear_fault_data():
    # # load motor data
    # motor_fn = "../data/tooth_fracture_data/1500rpm_CT_failure_6%_GP5_0_motor.csv"
    # motor_data = np.loadtxt(motor_fn, delimiter=",", skiprows=1)

    # time_motor = motor_data[:,0]
    # set_motor = motor_data[:,1]
    # meas_motor = motor_data[:,2]
    # set_propeller = motor_data[:,5]
    # meas_propeller = motor_data[:,6]

    # plt.plot(time_motor, set_propeller)
    # plt.show()

    # load sensor data and low pass filter
    sensor_fn = "../data/tooth_fracture_data/1500rpm_CT_failure_6%_GP5_0.csv"
    sensor_data = np.loadtxt(sensor_fn, delimiter=",", skiprows=1)

    time_sensor = sensor_data[:,0]-sensor_data[:,0][0]
    enc1_time = sensor_data[:,2]-sensor_data[:,2][0]
    enc1_angle = sensor_data[:,1]*np.pi/180  # convert degrees to radians
    speed1 = np.gradient(enc1_angle[10000:-10000], enc1_time[10000:-10000])  # cut data where measurement is in standstill
    speed1 = low_pass_filter(speed1, 300, 3012)
    enc2_time = sensor_data[:,4]-sensor_data[:,4][0]
    enc2_angle = sensor_data[:,3]*np.pi/180  # convert degrees to radians
    speed2 = np.gradient(enc2_angle[10000:-10000], enc2_time[10000:-10000])
    speed2 = low_pass_filter(speed2, 300, 3012)
    print(speed2)
    plt.plot(enc2_time[10000:-10000], speed2)
    plt.show()
    ciao

    torque1 = low_pass_filter(sensor_data[:,-1], 300, 3012)
    torque2 = low_pass_filter(sensor_data[:,-2], 300, 3012)

    plt.plot(time_sensor, speed1)
    # plt.plot(time_sensor, speed2)
    plt.show()

    # resample sensor data to 1kHz

    # save data
    # np.savetxt(
    #     "../data/masters_data/processed_data/no_load_2000_sensor.csv",
    #     measurements,
    #     delimiter=","
    # )
    # np.savetxt(
    #     "../data/masters_data/processed_data/no_load_2000_motor.csv",
    #     motor_measurements,
    #     delimiter=","
    # )

    return


if __name__ == "__main__":
    process_gear_fault_data()
