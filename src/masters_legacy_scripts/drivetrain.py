import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim, dimpulse, lti, lsim, impulse
import pickle
import opentorsion as ot

def step_excitation(step_start, step_end, step_value, initial_value=0):
    '''
    A step input function.
    '''

    return lambda t: initial_value + (t >= step_start)*(t < step_end)*step_value

def sinusoidal_excitation(omegas, amplitudes, offset=0, phase=0):
    """
    Sinusoidal excitation function.
    A sum of sine waves.

    Parameters
    omegas : list of floats
        Excitation frequencies (rad/s)
    amplitudes : list of floats
        Excitation amplitudes (Nm)
    offset : float
        DC component of the excitation (Nm)
    phase : float
        Excitation phase

    Returns
    excitation_function : lambda
    """

    return lambda t: offset + sum([amplitudes[i]*0.5*np.sin(omegas[i]*t + phase) for i in range(len(amplitudes))])

def excitation_matrix(t, load, dof):
    '''
    Excitation in matrix form. This function assumes the load is always applied to the last node of the drivetrain.

    Parameters
    t : ndarray
        timesteps
    load : lambda function
        excitation function
    dof : int
        number of degrees of freedom of the drivetrain
    '''
    U = np.zeros(t.shape[0])
    for i in range(1, dof):
        U = np.vstack([U, np.zeros(t.shape[0])])

    U[-1,:] += load(t)

    return U

def drivetrain_3dof():
    """
    Mechanical drivetrain as an openTorsion assembly instance.

    Returns:
        assembly: opentorsion assembly instance
            A 3-DOF mechanical drivetrain modeled as lumped masses and flexible shafts
            (lumped mass - shaft - lumped mass - shaft - lumped mass).
    """
    # Disk 1 inertia
    J1 = 0.8
    # Disk 2 inertia
    J2 = 0.5
    # Disk 3 inertia
    J3 = 0.7
    # Shaft 1 stiffness
    k1 = 1.5e4
    # Shaft 2 stiffness
    k2 = 1e4

    disks, shafts = [], []
    shafts.append(ot.Shaft(0, 1, None, None, k=k1, I=0))
    shafts.append(ot.Shaft(1, 2, None, None, k=k2, I=0))
    disks.append(ot.Disk(0, I=J1))
    disks.append(ot.Disk(1, I=J2))
    disks.append(ot.Disk(2, I=J3))
    assembly = ot.Assembly(shafts, disk_elements=disks)
    _, f, _ = assembly.modal_analysis()

    return assembly

def manually_built_3dof():
    """
    Mechanical drivetrain with manually constructed state matrices.

    Returns:
        assembly: opentorsion assembly instance
            A 3-DOF mechanical drivetrain modeled as lumped masses and flexible shafts
            (lumped mass - shaft - lumped mass - shaft - lumped mass).
    """
    # Disk 1 inertia
    J1 = 0.8
    # Disk 2 inertia
    J2 = 0.5
    # Disk 3 inertia
    J3 = 0.7
    # Shaft 1 stiffness
    k1 = 1.5e4
    # Shaft 2 stiffness
    k2 = 1e4

    return M, C, K

def testbench():
    '''
    Kongsberg testbench openTorsion model.
    '''
    shafts = []
    disks = []
    gears = []

    disks.append(ot.Disk(0, I=6.5e-4))
    shafts.append(ot.Shaft(0, 1, 0, 0, k=1.9039e5 , I=1.4420e-4, c=8.0804)) # driving motor, coupling
    shafts.append(ot.Shaft(1, 2, 0, 0, k=6.9487e3, I=3.7880e-6, c=0.2949)) # shaft
    shafts.append(ot.Shaft(2, 3, 0, 0, k=90, I=3e-6, c=0.2387)) # elastomer coupling hub
    shafts.append(ot.Shaft(3, 4, 0, 0, k=90, I=2e-6, c=0.2387)) # elastomer coupling middle piece
    shafts.append(ot.Shaft(4, 5, 0, 0, k=90, I=0, c=0.2387)) # elastomer coupling hubs & shaft
    disks.append(ot.Disk(5, I=7.8091e-3))
    shafts.append(ot.Shaft(5, 6, 0, 0, k=90, I=2e-6, c=0.2387))# elastomer coupling middle piece
    shafts.append(ot.Shaft(6, 7, 0, 0, k=90, I=0, c=0.0013)) # elastomer coupling hub & shaft
    shafts.append(ot.Shaft(7, 8, 0.342e3, 0.008e3, G=80e9, rho=7800)) # new shaft (shaft & coupling)
    # shafts.append(Shaft(7, 8, 0, 0, k=4.19e4, I=(5.0171e-5+3.1708e-6), c=1.7783)) # old shaft (shaft & coupling)
    shafts.append(ot.Shaft(8, 9, 0, 0, k=5.4e3, I=6.5e-6, c=0.2292)) # torque transducer
    shafts.append(ot.Shaft(9, 10, 0, 0, k=4.19e4, I=5.65e-5, c=1.7783)) # torque transducer & coupling
    shafts.append(ot.Shaft(10, 11, 0, 0, k=1.2192e3, I=4.2685e-6, c=0.5175)) # shaft
    gear1 = ot.Gear(11, I=3.2467e-4, R=1)
    gears.append(gear1) # shaft & gear
    gears.append(ot.Gear(12, I=0, R=3, parent=gear1))
    shafts.append(ot.Shaft(12, 13, 0, 0, k=3.1e4, I=1.2e-4, c=1.3157)) # coupling
    shafts.append(ot.Shaft(13, 14, 0, 0, k=1.1429e3, I=1.1516e-5, c=0.0485)) # shaft
    shafts.append(ot.Shaft(14, 15, 0, 0, k=3.1e4, I=1.3152e-4, c=1.3157)) # shaft & coupling
    shafts.append(ot.Shaft(15, 16, 0, 0, k=1.2192e4, I=4.2685e-6, c=0.5175)) # shaft
    gear2 = ot.Gear(16, I=2.6927e-4, R=1)
    gears.append(gear2) # shaft & gear
    gears.append(ot.Gear(17, I=0, R=4, parent=gear2))
    shafts.append(ot.Shaft(17, 18, 0, 0, k=1.38e5, I=1.8e-4, c=5.8569)) # coupling
    shafts.append(ot.Shaft(18, 19, 0, 0, k=2e4, I=2e-5, c=0.8488)) # torque transducer
    shafts.append(ot.Shaft(19, 20, 0, 0, k=1.38e5, I=2e-4, c=5.8569)) # torque trandsucer & coupling
    shafts.append(ot.Shaft(20, 21, 0, 0, k=1.2192e4, I=4.2685e-6, c=0.5175)) # shaft
    # disks.append(Disk(21, I=7.8e-3)) # shaft, mass, planetary gear & load generator
    disks.append(ot.Disk(21, I=4.9535e-2)) # shaft, mass, planetary gear & load generator

    assembly = ot.Assembly(shafts, disk_elements=disks, gear_elements=gears)
    _, f, _ = assembly.modal_analysis()
    print(f.round(2))

    return assembly

def state_matrices(assembly):
    """
    Create state-space matrices A and B of an openTorsion assembly.

    Parameters:
        assembly : openTorsion assembly instance
            Mechanical drivetrain model

    Returns:
        A : numpy.ndarray
            The state matrix
        B : numpy.ndarray
            The input matrix
    """
    M, K = assembly.M(), assembly.K()  # Mass and stiffness matrices
    C = assembly.C_modal(M, K, xi=0.02)  # Modal damping matrix, modal damping coefficient 0.02 used
    Z = np.zeros(M.shape)
    I = np.eye(M.shape[0])
    M_inv = LA.inv(M)

    A = np.vstack([np.hstack([Z, I]), np.hstack([-M_inv @ K, -M_inv @ C])])

    B = np.vstack([Z, M_inv])

    return A, B

def dlti_system(A, B, C, D, dt=1):
    '''
    Returns an instance of a discrete LTI-system using the state-space matrices.
    '''

    return dlti(A, B, C, D, dt=dt)

def lti_system(A, B, C, D):
    '''
    Returns an instance of a continuous LTI-system using the state-space matrices.
    '''

    return lti(A, B, C, D)

def noisy_simulation(time, A, B, C, U):
    '''
    A simulation with process and measurement noise.

    Returns the system output.
    '''
    dt = np.mean(np.diff(time))
    ## add gaussian white noise to measurement and process ##
    R = 1e-3*np.eye(A.shape[0]) # measurement covariance
    Q = 1e-2*np.eye(A.shape[0]) # process covariance
    r = np.random.multivariate_normal(np.zeros(R.shape[0]), R, time.shape[0]).T
    q = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, time.shape[0]).T

    x = np.zeros((A.shape[0], 1))
    y = np.zeros((A.shape[0], 1))

    for i in range(1, time.shape[0]):
        x_new = A @ x + (B @ U[:,i]).reshape(B.shape[0], 1) + q[:,i].reshape(q.shape[0], 1)
        y = np.hstack((y, C @ x_new + r[:,i].reshape(r.shape[0], 1)))
        x = x_new

    return time, y

def simulated_experiment(show_plot=False, pickle_data=False):
    '''
    Simulation of a 3-DOF mechanical drivetrain excited with a sinusoidal excitation.
    '''
    assembly = drivetrain_3dof()
    A, B = state_matrices(assembly)
    C = np.eye(B.shape[0])
    D = np.zeros(B.shape)

    t = np.arange(0, 100.01, 0.01)
    dt = np.mean(np.diff(t))

    load = step_excitation(20, 21, 100)

    U = excitation_matrix(t, load, dof=assembly.dofs)

    ## continuous time simulation ##
    sys = lti_system(A, B, C, D)
    tout, yout, xout = lsim(sys, U.T, t)

    if show_plot:
        plt.plot(tout, yout[:,-1], label='continuous')
        plt.ylim(0, 5000)
        plt.legend()
        plt.show()

    if pickle_data:
        filename = '../data/drivetrain_simulation.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump([t, U, tout, yout], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    assembly = testbench()
    # simulated_experiment(show_plot=True, pickle_data=False)
