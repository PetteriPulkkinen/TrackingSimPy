import numpy as np
import filterpy.common


def constant_turn_rate_matrix(w, dt, dim=2, order=1):
    assert (dim == 2)
    assert (order == 1)

    F = np.array([
        [1, np.sin(w * dt) / w, 0, (np.cos(w * dt) - 1) / w],
        [0, np.cos(w * dt), 0, -np.sin(w * dt)],
        [0, (1 - np.cos(w * dt)) / w, 1, np.sin(w * dt) / w],
        [0, np.sin(w * dt), 0, np.cos(w * dt)]
    ])
    return F


"""
def constant_turn_rate_covariance(w, dt, dim=2, order=1):
    assert dim == 2
    assert order == 1

    sigma_y = 1
    sigma_x = 1
    q1 = np.array({
        (3 * w * dt - 4 * np.sin(w * dt) + 1 / 2 * np.sin(2 * w * dt)) * sigma_y ** 2 /
        (2 * w ** 3) + (w * dt - 1 / 2 * np.sin(2 * w * dt)) * sigma_x ** 2 / (2 * w ** 3),
        # ------
        -(2 * np.cos(w * dt) - 1 - np.cos(w * dt) ** 2) * sigma_y ** 2 / (2 * w ** 2) + np.sin(
            w * dt) ** 2 * sigma_x ** 2 / (2 * w ** 2),
        # ------
        (2 * np.cos(w * dt) - 1 - np.cos(w * dt) ** 2) * sigma_y ** 2 / (2 * w ** 2)

    })

    Q = np.zeros(shape=(4, 4))
    return Q
"""


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
