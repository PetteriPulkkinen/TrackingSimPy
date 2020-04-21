import numpy as np


def constant_turn_rate_matrix(w, dt):
    """
    f = np.array([
        [1, w**(-1)*np.sin(w*dt), w**(-2)*(1 - np.cos(w * dt))],
        [0, np.cos(w * dt), w**(-1)*np.sin(w*dt)],
        [0, -w*np.sin(w*dt), np.cos(w*dt)]
    ])
    z = np.zeros((3, 3))
    F = np.concatenate((np.concatenate((f, z), axis=0), np.concatenate((z, f), axis=0)), axis=1)
    """

    F = np.array([
        [1, np.sin(w * dt)/w, 0, 0, -(1 - np.cos(w * dt)) / w, 0],
        [0, np.cos(w * dt), 0, 0, -np.sin(w * dt), 0],
        [0, 0, 0, 0, 0, 0],
        [0, (1 - np.cos(w * dt)) / w, 0, 1, np.sin(w * dt) / w, 0],
        [0, np.sin(w * dt), 0, 0, np.cos(w * dt), 0],
        [0, 0, 0, 0, 0, 0],
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


def singer_process_matrix(dt, corr_acc):
    return np.array([
            [1, dt, (corr_acc*dt - 1 + np.exp(-corr_acc*dt) / np.power(corr_acc, 2))],
            [0, 1 , (1 - np.exp(-corr_acc*dt)/corr_acc)],
            [0, 0 , np.exp(-corr_acc*dt)],
        ])


def singer_process_covariance(dt, corr_acc, std_acc):
    return 2*corr_acc*std_acc* np.array([
            [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
            [dt ** 4 / 8 , dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 3 / 6 , dt ** 2 / 2, dt         ]
        ])

