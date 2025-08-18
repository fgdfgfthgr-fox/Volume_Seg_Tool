import numpy as np
import numba


@numba.njit
def welford_mean_std(arr, where=None):
    """Calculate mean and standard deviation of a Numpy array using the Welford algorithm.

    Args:
        arr (np.ndarray): Input array.
        where (np.ndarray, optional): Boolean array of the same shape as `arr` indicating
                                      which elements to include. Defaults to None (include all).

    Returns:
        tuple: (mean, std) as float64 values. Returns (nan, nan) if no elements are included.
    """
    arr_flat = arr.ravel()
    if where is not None:
        where_flat = where.ravel()
    else:
        where_flat = None

    count = 0
    mean = 0.0
    M2 = 0.0

    n = arr_flat.size
    for i in range(n):
        if where_flat is not None and not where_flat[i]:
            continue
        x = arr_flat[i]
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2

    if count == 0:
        return (np.nan, np.nan)
    elif count == 1:
        return (mean, np.nan)
    else:
        variance = M2 / (count - 1)
        std = np.sqrt(variance)
        return (mean, std)
