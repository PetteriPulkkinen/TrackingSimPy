import numpy as np


def angle_in_2D(x, y):
    """Calculate 2D angle in radians from x-axis [0, 2*pi].

    Args:
        x: x-coordinate
        y: y-coordinate
    """
    if x >= 0 and y >= 0:  # 1st quadrant
        return np.arctan2(y, x)
    elif x < 0 and y >= 0:  # 2nd quadrant
        return np.pi/2 + np.arctan2(np.abs(x), np.abs(y))
    elif x < 0 and y < 0:  # 3rd quadrant
        return np.pi + np.arctan2(np.abs(y), np.abs(x))
    else:  # 4th quadrant
        return 3/2*np.pi + np.arctan2(np.abs(x), np.abs(y))


def angle_error_in_2D(alpha, beta):
    """Calculate minimum 2D angle in radians e.g. (pi/4, 7/4 pi) -> pi/4

    alpha: first angle
    beta: second angle
    """
    assert(alpha < 2*np.pi)
    assert(beta < 2*np.pi)

    error_init = np.abs(alpha - beta)

    if error_init <= np.pi:
        return error_init
    else:
        return 2*np.pi - np.max([alpha, beta]) + np.min([alpha, beta])


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
    """Calculate angle error straight from the position tuples.

    :param pos1: Tuple of x and y coordinates
    :param pos2: Tuple of x and y coordinates.
    :return: angle error in a floating point number
    """
    r1 = pos_to_radius_2D(pos1)
    r2 = pos_to_radius_2D(pos2)

    return np.abs(r1-r2)
