from filterpy.kalman import IMMEstimator
import numpy as np
from trackingsimpy.common.trigonometrics import pos_to_angle_error_2D


def normalize_innovation(y, R):
    w, v = np.linalg.eig(R)
    return (np.linalg.inv(np.sqrt(np.diag(w))) @ np.linalg.inv(v) @ y).flatten()


class TrackingComputer(object):
    """This class combines all objects needed to propagate number of cycles to achieve
    one update step. At the moment only single target scenarios are considered.
    """
    def __init__(self, tracker, radar, n_max, alpha=0.998, x0=None, P0=None):
        """
        Args:
            tracker: FilterPy style tracker filter
            radar: Radar to be used to observe the target
            n_max: Number of subsequent illuminations until track is considered lost
            alpha: Discount factor for past observations [0, 1)
        """
        self.P0 = P0
        self.x0 = x0
        self.tracker = tracker
        self.radar = radar
        self.n_max = n_max
        self.alpha = alpha
        self._discount = 1
        self.z = None  # last measurement
        self.y = None  # last innovation
        self.theta = None  # last angle error measurement
        self.theta_smoothed = None  # last smoothed angle error
        self.yn = None  # last normalized innovation
        self.yn_smoothed = None  # smoothed normalized innovations
        self.R_est = None  # last estimated measurement matrix
        self.n_missed = None  # last number of missed detections
        self.snr = None  # last snr
        self.current_time = 0

    def initialize(self, x0=None, P0=None):
        """Initialize tracking computer before starting the tracking task.

        Args:
            x0: Initial state estimate
            P0: Initial covariance estimate
        """
        if x0 is None:
            x0 = self.x0

        if P0 is None:
            P0 = self.P0

        # reset radar here?
        self.current_time = 0
        self.snr = self.radar.sn0
        self.y = np.zeros(self.radar.H.shape[0])
        self.R_est = None
        self.yn_smoothed = self.radar.H @ x0
        self.theta_smoothed = 0
        self.theta = 0

        # A little bit of dirty hack because IMMEstimator works differently than Kalman filter
        if type(self.tracker) == IMMEstimator:
            for filt in self.tracker.filters:
                filt.x = x0
                filt.P = P0
        else:
            self.tracker.x = x0
            self.tracker.P = P0

        self._reset_cycle()

    def _reset_cycle(self):
        self._discount = 1

    def predict(self):
        """Predict a priori estimates using tracker"""
        self._discount *= self.alpha
        self.current_time += 1
        self.tracker.predict()

    def update_track(self):
        """
        Returns:
            (True if track update was successful, False otherwise.;
             Number of missed detections, False otherwise.)
        """
        # initialize variables
        n_missed = 0
        update_successful = True

        while True:
            detection_occurred, self.z, self.R_est, self.snr = self.radar.illuminate(self.tracker.x_prior)

            if detection_occurred:
                break
            else:
                n_missed += 1

            if n_missed >= self.n_max:
                update_successful = False
                break

        if detection_occurred:
            # Calculate Innovation
            pos_est = self.radar.H @ self.tracker.x_prior.flatten()
            pos_meas = self.z.flatten()
            self.y = pos_meas - pos_est

            # Calculate absolute value of innovation in angle
            self.theta = pos_to_angle_error_2D(pos_meas, pos_est)
            self.theta_smoothed = 0.9*self.theta_smoothed + 0.1*self.theta

            # Calculate normalized innovation
            self.yn = normalize_innovation(self.y, R=self.R_est)
            self.yn_smoothed = (1 - self._discount) * self.yn + self._discount * self.yn_smoothed
            # Update tracker using the measurement
            if type(self.tracker) == IMMEstimator:
                for filt in self.tracker.filters:
                    filt.R = self.R_est
                    self.tracker.update(self.z)
            else:
                self.tracker.update(self.z, R=self.R_est)
            self._reset_cycle()

        self.n_missed = n_missed
        return update_successful, n_missed
