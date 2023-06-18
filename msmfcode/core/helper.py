import numpy as np


def mod(x, y, decimals=8):
    """
    Computes the remainder of the division x/y without floating point errors. The result of the division is rounded to
    the given number of decimals.
    :param float x: Dividend
    :param float y: Divisor
    :param int decimals: Number of decimals the division is rounded to
    :return:
    """
    fixed_float = np.round(x/y, decimals)
    return fixed_float - np.floor(fixed_float)
