import numba
import numpy as np


@numba.njit(parallel=True, fastmath=True)
def interpolant(t):
    return t ** 3 * (t * (t * 6 - 15) + 10)


@numba.njit(parallel=True, fastmath=True)
def generate_perlin_noise_3d(shape, res, tileable=(False, False, False)):
    """Based on:
    https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin3d.py
    and
    https://github.com/pvigier/perlin-numpy/issues/9#issue-968667149
    """
    dtype = np.float32
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])

    range1 = np.arange(0, res[0], delta[0]).astype(dtype) % 1
    range2 = np.arange(0, res[1], delta[1]).astype(dtype) % 1
    range3 = np.arange(0, res[2], delta[2]).astype(dtype) % 1

    grid = np.empty(shape=(shape[0], shape[1], shape[2], 3), dtype=dtype)
    # grid -> [shape[0], shape[1], shape[2], 3]

    for idx in numba.prange(shape[0]):
        grid[idx, :, :, 0] = range1[idx]

    for idx in numba.prange(shape[1]):
        grid[:, idx, :, 1] = range2[idx]

    for idx in numba.prange(shape[2]):
        grid[:, :, idx, 2] = range3[idx]

    # Gradients
    theta = 2 * np.pi * \
        np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1).astype(dtype)
    phi = 2 * np.pi * \
        np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1).astype(dtype)

    gradients = np.stack(
        (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)),
        axis=-1
    )
    # gradients -> [res[0] + 1, res[1] + 1, res[2] + 1, 3]

    if tileable[0]:
        gradients[-1, :, :] = gradients[0, :, :]
    if tileable[1]:
        gradients[:, -1, :] = gradients[:, 0, :]
    if tileable[2]:
        gradients[:, :, -1] = gradients[:, :, 0]

    grad_shape = (
        d[0] * gradients.shape[0], d[1] * gradients.shape[1],
        d[2] * gradients.shape[2], 3)
    grad_matrix = np.empty(shape=grad_shape, dtype=dtype)

    for idx1 in numba.prange(gradients.shape[0]):
        for idx2 in numba.prange(gradients.shape[1]):
            for idx3 in numba.prange(gradients.shape[2]):
                grad_matrix[
                    d[0] * idx1: d[0] * (idx1 + 1),
                    d[1] * idx2: d[1] * (idx2 + 1),
                    d[2] * idx3: d[2] * (idx3 + 1),
                ] = gradients[idx1, idx2, idx3]

    gradients = grad_matrix
    # gradients -> [shape[0] + d[0], shape[1] + d[1], shape[2] + d[2], 3]

    g000 = gradients[:-d[0], :-d[1], :-d[2]]
    g100 = gradients[d[0]:, :-d[1], :-d[2]]
    g010 = gradients[:-d[0], d[1]:, :-d[2]]
    g110 = gradients[d[0]:, d[1]:, :-d[2]]
    g001 = gradients[:-d[0], :-d[1], d[2]:]
    g101 = gradients[d[0]:, :-d[1], d[2]:]
    g011 = gradients[:-d[0], d[1]:, d[2]:]
    g111 = gradients[d[0]:, d[1]:, d[2]:]
    # gxy -> [shape[0], shape[1], shape[2], 3]

    # Ramps

    n_bits = 3
    len_ = 2 ** n_bits
    code = ((np.arange(len_).reshape(len_, 1) & (1 << np.arange(n_bits)))) > 0
    code = code.astype(np.int32)
    # gradients -> [8, 3]

    n000 = np.sum((grid - code[0]) * g000, 3)
    n100 = np.sum((grid - code[1]) * g100, 3)
    n010 = np.sum((grid - code[2]) * g010, 3)
    n110 = np.sum((grid - code[3]) * g110, 3)
    n001 = np.sum((grid - code[4]) * g001, 3)
    n101 = np.sum((grid - code[5]) * g101, 3)
    n011 = np.sum((grid - code[6]) * g011, 3)
    n111 = np.sum((grid - code[7]) * g111, 3)
    # nxyz -> [shape[0], shape[1], shape[2]]

    t = interpolant(grid)
    t1 = 1 - t[:, :, :, 0]

    n00 = t1 * n000 + t[:, :, :, 0] * n100
    n10 = t1 * n010 + t[:, :, :, 0] * n110
    n01 = t1 * n001 + t[:, :, :, 0] * n101
    n11 = t1 * n011 + t[:, :, :, 0] * n111

    t2 = 1 - t[:, :, :, 1]
    n0 = t2 * n00 + t[:, :, :, 1] * n10
    n1 = t2 * n01 + t[:, :, :, 1] * n11

    output = (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1

    return output