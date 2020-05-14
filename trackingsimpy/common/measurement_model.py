import numpy as np

from trackingsimpy.common.trigonometrics import rotmat_2D


def position_measurement_matrix(dim, order):
    """Measurement matrix for position measurements."""
    H = np.zeros(shape=(dim, dim*(order+1)))
    for idx in range(dim):
        H[idx, idx*(order+1)] = 1
    return H


def meas_acovmat_2D(distance, r_std, theta_std, angle):
    T = rotmat_2D(angle)
    R = np.array([
        [r_std**2, 0],
        [0, (distance * theta_std)**2]
    ])
    return T @ R @ T.T
