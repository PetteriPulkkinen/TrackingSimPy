import numpy as np
from collections import namedtuple


def measurement_matrix(distance, r_std, theta_std, angle):
    T = rotation_matrix(angle)
    R = np.array([
        [r_std**2, 0],
        [0, (distance * theta_std)**2]
    ])
    return T @ R @ T.T


def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), - np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


def radar_snr(power, transmitter_gain, receiver_gain, wavelength, rcs, noise, distance):
    return power * transmitter_gain * receiver_gain * wavelength**2 * rcs / \
           ((4*np.pi)**3 * distance ** 4 * noise)


def snr_with_beam_losses(SN0, angular_error, beamwidth):
    return SN0 * np.exp(-2 * angular_error**2 / (beamwidth**2))


def radial_std(snr):
    return 20.18*np.sqrt(1 / np.sqrt(snr))


def angular_std(snr):
    return 2.18e-2*np.sqrt(1 / np.sqrt(snr))


def detection_probability(snr, pf):
    return np.power(pf, 1/(1 + snr))


def angle_in_2D(x, y):
    if x >= 0 and y >= 0:  # 1st quadrant
        return np.arctan2(y, x)
    elif x < 0 and y >= 0:  # 2nd quadrant
        return np.pi/2 + np.arctan2(np.abs(x), np.abs(y))
    elif x < 0 and y < 0:  # 3rd quadrant
        return np.pi + np.arctan2(np.abs(y), np.abs(x))
    else:  # 4th quadrant
        return 3/2*np.pi + np.arctan2(np.abs(x), np.abs(y))


def angle_error_in_2D(alpha, beta):
    assert(alpha < 2*np.pi)
    assert(beta < 2*np.pi)

    error_init = np.abs(alpha - beta)

    if error_init <= np.pi:
        return error_init
    else:
        return 2*np.pi - np.max([alpha, beta]) + np.min([alpha, beta])


RadarMeasurement = namedtuple('radar_measurement', ['z', 'R', 'R_est', 'angular_error', 'SNR', 'n_dwells'])


class Radar2D(object):
    def __init__(
            self,
            sn0,
            beamwidth,
            prob_f,
            order,
            ):
        """Radar that measures position in 2-dimensional coordinates.
        The measurement noise is spanned along radial and angular coordinates.

        :param sn0: The signal-to-noise ratio at central beam.
        :param prob_f: Probability of false alarm
        :param beamwidth: Main lobe -3dB beamwidth in radians
        :param order: State order (order 0 := position, order 1 := velocity, order 2:= acceleration)
        """
        self.dim = 2
        self.order = order
        self.n = self.dim

        self.sn0 = sn0
        self.pf = prob_f
        self.beamwidth = beamwidth

        self.H = np.zeros((self.dim, (order+1)*self.dim))
        self.H[0, 0] = 1
        self.H[1, self.order + 1] = 1

    def measure(self, target, tracker):
        """
        :param target: Target object
        :param tracker: Tracker object
        :return: Measurement of the target state corrupted with white noise
        """
        pos = (self.H @ target.x).flatten()
        pos_hat = (self.H @ tracker.x).flatten()

        distance = np.linalg.norm(pos)
        distance_est = np.linalg.norm(pos_hat)

        angle = angle_in_2D(pos[0], pos[1])
        angle_hat = angle_in_2D(pos_hat[0], pos_hat[1])

        angular_error = angle_error_in_2D(angle, angle_hat)

        snr = snr_with_beam_losses(self.sn0, angular_error, self.beamwidth)

        R = measurement_matrix(distance, radial_std(snr), angular_std(snr), angle)
        R_est = measurement_matrix(distance_est, radial_std(self.sn0), angular_std(self.sn0), angle_hat)

        y = self.H @ target.x
        z = y + np.random.multivariate_normal(np.zeros(y.size), R).reshape(-1, 1)
        pd = detection_probability(snr, self.pf)
        n_dwells = 1 / pd

        measurement = RadarMeasurement(z.flatten(), R, R_est, angular_error, snr, n_dwells)
        return measurement

