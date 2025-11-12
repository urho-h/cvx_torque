import numpy as np
import scipy.linalg as LA

import opentorsion as ot


def testbench_powertrain():
    inertias = np.array([7.94e-4, 3.79e-6, 3.00e-6, 2.00e-6, 7.81e-3, 2.00e-6, 3.29e-6, 5.01e-5, 6.50e-6, 5.65e-5, 4.27e-6, 3.25e-4, 1.20e-4, 1.15e-5, 1.32e-4, 4.27e-6, 2.69e-4, 1.80e-4, 2.00e-5, 2.00e-4, 4.27e-6, 4.95e-2])

    stiffnesses = np.array([1.90e5, 6.95e3, 90.00, 90.00, 90.00, 90.00, 94.064, 4.19e4, 5.40e3, 4.19e4, 1.22e3, 4.33e4, 3.10e4, 1.14e3, 3.10e4, 1.22e4, 4.43e4, 1.38e5, 2.00e4, 1.38e5, 1.22e4])

    damping = np.array([8.08, 0.29, 0.24, 0.24, 0.24, 0.24, 0.00, 1.78, 0.23, 1.78, 0.52, 1.84, 1.32, 0.05, 1.32, 0.52, 1.88, 5.86, 0.85, 5.86, 0.52])

    external_damping = np.array([0.0030, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0042, 0, 0, 0, 0, 0.0042, 0, 0, 0, 0, 0.2400])

    shafts, disks, gears = [], [], []

    disks.append(ot.Disk(0, I=inertias[0], c=external_damping[0]))
    shafts.append(ot.Shaft(0, 1, k=stiffnesses[0], c=damping[0]))
    disks.append(ot.Disk(1, I=inertias[1], c=external_damping[1]))
    shafts.append(ot.Shaft(1, 2, k=stiffnesses[1], c=damping[1]))
    disks.append(ot.Disk(2, I=inertias[2], c=external_damping[2]))
    shafts.append(ot.Shaft(2, 3, k=stiffnesses[2], c=damping[2]))
    disks.append(ot.Disk(3, I=inertias[3], c=external_damping[3]))
    shafts.append(ot.Shaft(3, 4, k=stiffnesses[3], c=damping[3]))
    disks.append(ot.Disk(4, I=inertias[4], c=external_damping[4]))
    shafts.append(ot.Shaft(4, 5, k=stiffnesses[4], c=damping[4]))
    disks.append(ot.Disk(5, I=inertias[5], c=external_damping[5]))
    shafts.append(ot.Shaft(5, 6, k=stiffnesses[5], c=damping[5]))
    disks.append(ot.Disk(6, I=inertias[6], c=external_damping[6]))
    shafts.append(ot.Shaft(6, 7, k=stiffnesses[6], c=damping[6]))
    disks.append(ot.Disk(7, I=inertias[7], c=external_damping[7]))
    shafts.append(ot.Shaft(7, 8, k=stiffnesses[7], c=damping[7]))
    disks.append(ot.Disk(8, I=inertias[8], c=external_damping[8]))
    shafts.append(ot.Shaft(8, 9, k=stiffnesses[8], c=damping[8]))
    disks.append(ot.Disk(9, I=inertias[9], c=external_damping[9]))
    shafts.append(ot.Shaft(9, 10, k=stiffnesses[9], c=damping[9]))
    disks.append(ot.Disk(10, I=inertias[10], c=external_damping[10]))
    shafts.append(ot.Shaft(10, 11, k=stiffnesses[10], c=damping[10]))
    disks.append(ot.Disk(11, I=inertias[11], c=external_damping[11]))
    gears.append(gear_up := ot.Gear(11, 0, R=1))

    gears.append(ot.Gear(12, I=0, R=3, parent=gear_up))
    shafts.append(ot.Shaft(12, 13, k=stiffnesses[11], c=damping[11]))
    disks.append(ot.Disk(13, I=inertias[12], c=external_damping[12]))
    shafts.append(ot.Shaft(13, 14, k=stiffnesses[12], c=damping[12]))
    disks.append(ot.Disk(14, I=inertias[13], c=external_damping[13]))
    shafts.append(ot.Shaft(14, 15, k=stiffnesses[13], c=damping[13]))
    disks.append(ot.Disk(15, I=inertias[14], c=external_damping[14]))
    shafts.append(ot.Shaft(15, 16, k=stiffnesses[14], c=damping[14]))
    disks.append(ot.Disk(16, I=inertias[15], c=external_damping[15]))
    shafts.append(ot.Shaft(16, 17, k=stiffnesses[15], c=damping[15]))
    disks.append(ot.Disk(17, I=inertias[16], c=external_damping[16]))
    gears.append(gear_low := ot.Gear(17, I=0, R=1))

    gears.append(ot.Gear(18, I=0, R=4, parent=gear_low))
    shafts.append(ot.Shaft(18, 19, k=stiffnesses[16], c=damping[16]))
    disks.append(ot.Disk(19, I=inertias[17], c=external_damping[17]))
    shafts.append(ot.Shaft(19, 20, k=stiffnesses[17], c=damping[17]))
    disks.append(ot.Disk(20, I=inertias[18], c=external_damping[18]))
    shafts.append(ot.Shaft(20, 21, k=stiffnesses[18], c=damping[18]))
    disks.append(ot.Disk(21, I=inertias[19], c=external_damping[19]))
    shafts.append(ot.Shaft(21, 22, k=stiffnesses[19], c=damping[19]))
    disks.append(ot.Disk(22, I=inertias[20], c=external_damping[20]))
    shafts.append(ot.Shaft(22, 23, k=stiffnesses[20], c=damping[20]))
    disks.append(ot.Disk(23, I=inertias[21], c=external_damping[21]))

    assembly = ot.Assembly(shaft_elements=shafts, disk_elements=disks, gear_elements=gears)
    _, f, _ = assembly.modal_analysis()
    # print("Eigenfrequencies (Hz): ", f/(2*np.pi))

    return assembly


def get_testbench_state_space(dt, use_motor=False):
    """
    This function returns the discrete-time state-space matrices of the testbench model.
    """
    assembly = testbench_powertrain()
    Ac, Bc, _, _ = assembly.state_space()
    X = assembly.X
    X_inv = LA.pinv(X)
    AcX = X @ Ac @ X_inv
    BcX_full = X @ Bc
    BcX = np.vstack((BcX_full[:,0], BcX_full[:,-1])).T

    A, B = assembly.continuous_2_discrete(AcX, BcX, dt)
    C = np.zeros((4, A.shape[0]))
    if use_motor:
        C[2,0] += 1
    else:
        C[2,8] += 1
    C[3,18] += 1
    C[0,27] += 1
    C[1,28] += 1
    D = np.zeros((C.shape[0], BcX.shape[1]))

    # C and D without propeller shaft torque
    c = np.zeros((3, A.shape[0]))
    if use_motor:
        c[2,0] += 1
    else:
        c[2,8] += 1
    c[0,27] += 1
    c[1,28] += 1
    d = np.zeros((c.shape[0], BcX.shape[1]))
    print(A.shape)

    return A, B, C, D, c, d


if __name__ == "__main__":
    assembly = testbench_powertrain()
    plot_tools = ot.Plots(assembly)
    plot_tools.plot_assembly()
    print(assembly)
    _, _, _, _, _ = get_testbench_state_space(1e-3, use_motor=False)
