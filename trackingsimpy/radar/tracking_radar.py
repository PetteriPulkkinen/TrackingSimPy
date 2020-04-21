from .generic_sensor import Sensor
from trackingsimpy.common.measurement_model import position_measurement_matrix
from .radar_2d import angle_in_2D, angle_error_in_2D, measurement_covariance_matrix
from .radar_2d import radial_std, snr_with_beam_losses, detection_probability
from .radar_2d import angular_std
import numpy as np


class TrackingRadar(Sensor):
    """Class intended to be used in a single target tracking scenarios."""
    def __init__(self, target, sn0, pfa, beamwidth, dim, order):
        """
        Args:
            target: Target to be tracked
            sn0: Center beam signal-to-noise ratio
            pfa: Probability of false alarm
            beamwidth: Main lobe -3dB beamwidth in radians
            order: State order (order 0 := position, order 1 := velocity, order 2:= acceleration)
        """
        if dim != 2:
            raise NotImplementedError

        H = position_measurement_matrix(dim, order)  # measure only position
        super(TrackingRadar, self).__init__(H=H, R=None)  # R will be set in real-time
        self.target = target
        self.beamwidth = beamwidth
        self.sn0 = sn0
        self.pfa = pfa

        # real-time operation parameters
        self.angle_error = None

    def illuminate(self, prediction):
        """
        Args:
            prediction: Predicted target state

        Returns:
            Bool for detection occurred, measurement and estimated measurement covariance
        """
        pos_est = (self.H @ prediction).flatten()
        pos = (self.H @ self.target.x).flatten()
        angle_est = angle_in_2D(pos_est[0], pos_est[1])
        angle = angle_in_2D(pos[0], pos[1])
        self.angle_error = angle_error_in_2D(angle_est, angle)

        snr = snr_with_beam_losses(self.sn0, self.angle_error, self.beamwidth)
        pd = detection_probability(snr, self.pfa)

        detection_occurred = bool(np.random.binomial(n=1, p=pd))

        # No detection occurred, so the radar returns without measurement
        if not detection_occurred:
            return (detection_occurred,
                    np.ones(self.H.shape[0])*np.inf,
                    np.ones((self.H.shape[0],)*2)*np.inf,
                    0)

        distance = np.linalg.norm(pos)

        R = measurement_covariance_matrix(
            distance, radial_std(snr), angular_std(snr), angle)

        z = super().measure(self.target.x, R=R)
        return detection_occurred, z, R, snr
