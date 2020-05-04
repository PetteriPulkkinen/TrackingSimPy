import numpy as np


def angle_in_2D(x, y):
    """Calculate 2D angle in radians from x-axis [-pi, pi].

    Args:
        x: x-coordinate
        y: y-coordinate
    """
    return np.arctan2(y, x)


def angle_error_in_2D(alpha, beta):
    """Calculate angle error in radians between [-pi, pi]

    alpha: first angle
    beta: second angle
    """
    assert(-np.pi <= alpha <= np.pi)
    assert(-np.pi <= alpha <= np.pi)

    error_init = beta - alpha

    if error_init > np.pi:
        return error_init - 2*np.pi
    elif error_init < -np.pi:
        return error_init + 2*np.pi
    else:
        return error_init


def pos_to_angle_2D(pos):
    """Calculate angle in radians [0, 2pi].

    Args:
        pos: 2D position coordinate (x, y)
    Returns:
        Angle as a floating point number
    """
    return angle_in_2D(pos[0], pos[1])


def pos_to_radius_2D(pos):
    """Calculate radius from 2d position array.

    Args:
        pos: 2D position coordinate (x, y)
    Returns:
        Radius as a floating point number
    """
    return np.linalg.norm(pos)


def pos_to_angle_error_2D(pos1, pos2):
    """Calculate angle error straight from the position tuples.

    :param pos1: Tuple of x and y coordinates
    :param pos2: Tuple of x and y coordinates.
    :return: angle error in a floating point number
    """
    alpha = pos_to_angle_2D(pos1)
    beta = pos_to_angle_2D(pos2)
    return angle_error_in_2D(alpha, beta)


def pos_to_radius_error_2D(pos1, pos2):
    """Calculate radius error straight from the position tuples.

    :param pos1: Tuple of x and y coordinates
    :param pos2: Tuple of x and y coordinates.
    :return: radius error as a floating point number
    """
    r1 = pos_to_radius_2D(pos1)
    r2 = pos_to_radius_2D(pos2)

    return r1-r2


def rotmat_2D(theta):
    return np.array([
        [np.cos(theta), - np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
