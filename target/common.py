import numpy as np


def constant_turn_rate_matrix(turn_rate, dt):
    F = np.array([
        [1, np.sin(turn_rate*dt)/turn_rate,     0, -(1-np.cos(turn_rate*dt))/turn_rate],
        [0, np.cos(turn_rate*dt),               0,               -np.sin(turn_rate*dt)],
        [0, (1-np.cos(turn_rate*dt))/turn_rate, 1,      np.sin(turn_rate*dt)/turn_rate],
        [0, np.sin(turn_rate*dt),               0,                np.cos(turn_rate*dt)]
    ])
    return F


def constant_velocity_matrix(dt):
    F = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])
    return F

