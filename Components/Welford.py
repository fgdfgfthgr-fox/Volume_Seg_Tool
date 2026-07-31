from numba import njit
import numpy as np


@njit
def welford_mean_std(arr):
    """
    Calculate mean and standard deviation of a Numpy array using the Welford algorithm. Doesn't include 0.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        tuple: (mean, std) as float64. Returns (nan, nan) if no elements included.
    """
    arr_flat = arr.ravel()
    n = arr_flat.size

    count = 0
    mean = 0.0
    M2 = 0.0

    for i in range(n):
        x = arr_flat[i]
        if x == 0:
            continue
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2

    if count == 0:
        return np.nan, np.nan
    elif count == 1:
        return mean, np.nan
    else:
        variance = M2 / (count - 1)
        return mean, np.sqrt(variance)