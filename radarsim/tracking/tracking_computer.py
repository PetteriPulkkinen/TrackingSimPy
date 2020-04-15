from filterpy.kalman import IMMEstimator
import numpy as np


class TrackingComputer(object):
    """This class combines all objects needed to propagate number of cycles to achieve
    one update step. At the moment only single target scenarios are considered.
    """
    def __init__(self, tracker, radar, n_max, save=False):
        """
        Args:
            tracker: FilterPy style tracker filter
            radar: Radar to be used to observe the target
            n_max: Number of subsequent illuminations until track is considered lost.
        """
        self.tracker = tracker
        self.radar = radar
        self.n_max = n_max
        self.z = None  # last measurement
        self.y = None  # last innovation
        self.yn = None  # last normalized innovation
        self.R_est = None  # last estimated measurement matrix
        self.n_missed = None  # last number of missed detections

    def initialize(self, x0, P0):
        """Initilize tracking computer before starting the tracking task.

        Args:
            x0: Initial state estimate
            P0: Initial covariance estimate
        """
        # reset radar here?

        self.y = None
        self.R_est = None

        # A little bit of dirty hack because IMMEstimator works differently than Kalman filter
        if type(self.tracker) == IMMEstimator:
            for filt in self.tracker.filters:
                filt.x = x0
                filt.P = P0
        else:
            self.tracker.x = x0
            self.tracker.P = P0

    def predict(self):
        """Predict a priori estimates using tracker"""
        self.tracker.predict()

    def update_track(self):
        """
        Returns:
            (True if track update was successful, False otherwise.,
             Number of missed detections, False otherwise.)
        """
        # initialize variables
        n_missed = 0
        update_succesful = True

        while(True):
            detection_occured, self.z, R_est = self.radar.illuminate(self.tracker.x_prior)

            if detection_occured:
                break
            else:
                n_missed += 1

            if n_missed >= self.n_max:
                update_succesful = False
                break

        if detection_occured:
            # Calculate Innovation
            self.y = self.z.flatten() - self.tracker.H @ self.tracker.x_prior.flatten()

            # Calculate normalized innovation
            self.R_est = R_est
            w, v = np.linalg.eig(R_est)
            self.yn = (np.linalg.inv(np.sqrt(np.diag(w))) @ np.linalg.inv(v) @ self.y).flatten()

            # Update tracker using the measurement
            self.tracker.update(self.z, R=R_est)

        self.n_missed = n_missed
        return update_succesful, n_missed
