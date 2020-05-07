import numpy as np


def detection_probability(snr, pf):
    return np.power(pf, 1/(1 + snr))


def radial_std(snr):
    return 20.18*np.sqrt(1 / np.sqrt(snr))


def radar_snr(power, transmitter_gain, receiver_gain, wavelength, rcs, noise, distance):
    return power * transmitter_gain * receiver_gain * wavelength**2 * rcs / \
           ((4*np.pi)**3 * distance ** 4 * noise)


def snr_with_beam_losses(SN0, angular_error, beamwidth):
    return SN0 * np.exp(-np.log(2) * angular_error**2 / (beamwidth**2))


def angular_std(snr):
    return 0.54e-2*np.sqrt(1 / np.sqrt(snr))
