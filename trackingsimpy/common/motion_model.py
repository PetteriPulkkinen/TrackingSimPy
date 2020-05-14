import numpy as np
import filterpy.common

np.seterr('raise')


def constant_turn_rate_matrix(w, dt, dim=2, order=1):
    assert (dim == 2)
    assert (order == 1)

    WT = w * dt
    SWT = np.sin(WT)
    CWT = np.cos(WT)

    if w > 1e-6:
        F = np.array([
            [1, SWT / w, 0, (CWT - 1) / w],
            [0, CWT, 0, -SWT],
            [0, (1 - CWT) / w, 1, SWT / w],
            [0, SWT, 0, CWT]
        ])
    else:
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])
    return F


def const_turn_jac(x, dt):
    W = x[-1]
    WT = W*dt
    CWT = np.cos(WT)
    SWT = np.sin(WT)
    x_ = x[1]
    y_ = x[3]
    f25 = (-x_ * SWT - y_ * CWT) * dt
    f45 = (x_ * CWT - y_ * SWT) * dt

    if W > 1e-6:
        f15 = ((WT * CWT - SWT) * x_ + (1 - CWT - WT * SWT) * y_) / (W ** 2)
        f35 = (WT * (x_ * SWT + y_ * CWT) - (x_ * (1 - CWT) + y_ * SWT)) / (W ** 2)
        F = np.array([
            [1, SWT/W, 0, -(1-CWT)/W, f15],
            [0, CWT, 0, -SWT, f25],
            [0, (1-CWT)/W, 1, SWT/W, f35],
            [0, SWT, 0, CWT, f45],
            [0, 0, 0, 0, 1]
        ])
    else:
        F = np.array([
            [1, dt, 0, 0, 0],
            [0, 1, 0, 0, f25],
            [0, 0, 1, dt, 0],
            [0, 0, 0, 1, f45],
            [0, 0, 0, 0, 1]
        ])
    return F


def const_turn(x, dt):
    W = x[-1]
    _x = x[:-1]
    F = constant_turn_rate_matrix(W, dt)
    _x = F @ _x
    return np.append(_x, W)


def acceleration_control_matrix(dt, dim):
    """Acceleration matrix which can be used to control constant velocity process."""
    order = 1
    b = np.array([1 / 2 * dt, dt])
    B = np.zeros(shape=(2 * dim, dim))
    for idx in range(dim):
        idx_low = idx * (order + 1)
        idx_high = (idx + 1) * (order + 1)
        B[idx_low:idx_high, idx] = b
    return B


def kinematic_state_transition(dt, order, dim):
    F = np.zeros(((order + 1) * dim,) * 2)
    for idx in range(dim):
        idx_low = idx * (order + 1)
        idx_high = (idx + 1) * (order + 1)
        F[idx_low:idx_high, idx_low:idx_high] = filterpy.common.kinematic_state_transition(order, dt)
    return F


def singer_process_matrix(dt, corr_acc):
    return np.array([
        [1, dt, (corr_acc * dt - 1 + np.exp(-corr_acc * dt) / np.power(corr_acc, 2))],
        [0, 1, (1 - np.exp(-corr_acc * dt) / corr_acc)],
        [0, 0, np.exp(-corr_acc * dt)],
    ])


def singer_process_covariance(dt, corr_acc, std_acc):
    return 2 * corr_acc * std_acc * np.array([
        [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
        [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
        [dt ** 3 / 6, dt ** 2 / 2, dt]
    ])
