'''
Script for handling data generated in the simulations used in the SIRM conference paper.
'''

import numpy as np
import pickle

def get_sirm_dataset(pathname):
    '''
    Unpickles a dataset containing the shaft torque and
    excitation data of a simulated electric drive, with
    open-loop control and 1480 RPM operating speed reference.
    The data was used in the SIRM conference paper.

    Returns:

    time : list
        simulation time in seconds
    thetas : list
        measured rotational values at each drivetrain node (rad)
    omegas : list
        measured mechanical rotating speed at each drivetrain node (rad/s)
    '''

    try:
        with open(pathname, 'rb') as handle:
            dataset = pickle.load(handle)
            time, theta, omega, motor, load = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4]
    except EOFError:
        print("corrupted dataset")

    theta = np.array(theta)
    omega = np.array(omega)
    motor = np.array(motor)
    load = np.array(load)

    return time, theta, omega, motor, load

def construct_measurement(theta, omega, motor, load, dof, n, t_start, KF=False):
    '''
    Builds the batch measurement matrix.

    Parameters:

    theta : ndarray, shape (i,)
        rotational angle measurement
    omega : ndarray, shape (i,)
        rotational speed measurement
    motor : ndarray, shape (i,)
        motor load
    load : ndarray, shape (i,)
        external load

    Returns:

    measurements : ndarray
        angles and speeds stacked
    load : ndarray
        input matrix
    '''
    measurements = np.vstack([theta[0+t_start,:].reshape(3,1), omega[0+t_start,:].reshape(3,1)])

    if KF:
        for i in range(1+t_start, n+t_start):
            measurements = np.hstack([measurements, np.vstack([theta[i,:].reshape(3,1), omega[i,:].reshape(3,1)])])
    else:
        for i in range(1+t_start, n+t_start):
            measurements = np.vstack([measurements, np.vstack([theta[i,:].reshape(3,1), omega[i,:].reshape(3,1)])])

    inputs = np.zeros((dof, n))
    inputs[0,:] = motor[t_start:n+t_start]
    inputs[-1,:] = -load[t_start:n+t_start]

    return measurements, inputs

if __name__ == "__main__":
    pathname = "../data/rpm1480.0.pickle"
    time, theta, omega, motor, load = get_dataset(pathname)
    dof, n, t_start = 3, 100, 500000
    meas, inputs = construct_measurement(theta, omega, motor, load, dof, n, t_start)
    print(meas.shape)
