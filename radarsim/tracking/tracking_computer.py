from filterpy.kalman import IMMEstimator


class TrackingComputer(object):
    """This class combines all objects needed to propagate number of cycles to achieve
    one update step. At the moment only single target scenarios are considered.
    """
    def __init__(self, tracker, radar, n_max):
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

    def initialize(self, x0, P0):
        """Initilize tracking computer before starting the tracking task.

        Args:
            x0: Initial state estimate
            P0: Initial covariance estimate
        """
        # reset tracker and radar here?
        self.current_time = 0  # E.g. first prediction is done for k=1
        self.update_time = None  # This is automatically changed in beginning of the propagation

        # A little bit of dirty hack because IMMEstimator works differently than Kalman filter
        if type(self.tracker) == IMMEstimator:
            for filt in self.tracker.filters:
                filt.x = x0
                filt.P = P0
        else:
            self.tracker.x = x0
            self.tracker.P = P0

    def cycle(self):
        """Propagate one computation cycle.

        Returns:
            True if track was updated, False otherwise.
        """
        self.current_time += 1

        # Update tracker predictions
        self.tracker.predict()

        # initialize boolen variable to be returned by the method
        track_updated = False

        # Check if it time to update the track
        if self.current_time >= self.update_time:

            detection_occured, z, R_est = self.radar.illuminate(self.tracker.x_prior)

            if detection_occured:
                self.tracker.update(z, R=R_est)
                track_updated = True

        return track_updated

    def propagate(self, revisit_interval):
        """Propagate multiple cycles as long as the track is updated.

        Returns:
            True if track is updated succesfully, False otherwise.
        """
        self.update_time = self.current_time + revisit_interval

        n_missed = 0
        while(True):
            track_updated = self.cycle()
            if track_updated:
                return True

            n_missed += int(not track_updated)

            # Check if target has been illuminated at least n_max times without detection
            if n_missed >= self.n_max:
                return False  # break the loop and stop tracking
