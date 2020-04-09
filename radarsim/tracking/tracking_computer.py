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
        self.update_time = None
        self.current_time = None
        self.current_time = 0
        self.update_time = 0
        self.n_max = n_max
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
        # reset tracker and radar here?
        self.current_time = 0  # E.g. first prediction is done for k=1
        self.update_time = None  # This is automatically changed in beginning of the propagation

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

    def set_update_time(self, revisit_interval):
        """Sets next track update to be at current time plus revisit interval.

        Args:
            revisit_interval: Number of time steps from current time step after track is updated
        """
        self.update_time = self.current_time + revisit_interval

    def cycle(self):
        """Propagate one computation cycle.

        Returns:
            (True if track update trial was obtained, False otherwise.,
             True if target detection occured, False otherwise)
        """
        self.current_time += 1

        # Update tracker predictions
        self.tracker.predict()

        # initialize boolen variable to be returned by the method
        track_update_trial = False
        detection_occured = False

        # Check if it time to update the track
        if self.current_time >= self.update_time:
            track_update_trial = True

            detection_occured, z, R_est = self.radar.illuminate(self.tracker.x_prior)

            if detection_occured:
                # Calculate Innovation
                self.y = z.flatten() - self.tracker.H @ self.tracker.x_prior.flatten()

                # Calculate normalized innovation
                self.R_est = R_est
                w, v = np.linalg.eig(R_est)
                self.yn = (np.linalg.inv(np.sqrt(np.diag(w))) @ np.linalg.inv(v) @ self.y).flatten()

                # Update tracker using the measurement
                self.tracker.update(z, R=R_est)

        return track_update_trial, detection_occured

    def propagate(self, revisit_interval):
        """Propagate multiple cycles as long as the track is updated.

        Returns:
            (True if track is updated succesfully, False otherwise,
             number of missed detection on update trials)
        """
        self.update_time = self.current_time + revisit_interval

        self.n_missed = 0
        while(True):
            track_update_trial, detection_occured = self.cycle()
            if track_update_trial:
                if detection_occured:
                    return True, self.n_missed
                else:
                    self.n_missed += 1

            # Check if target has been illuminated at least n_max times without detection
            if self.n_missed >= self.n_max:
                return False, self.n_missed  # break the loop and stop tracking
