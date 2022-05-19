from itertools import product
import numpy as np


def fast_fov(mask, pos, fov) -> np.array:
    """given a mask and a location in it, ensure that the fov around
    the position is marked as 1"""
    d_range = range(-fov, fov + 1)
    nr, nc = mask.shape
    d_fov = np.array(list(product(d_range, repeat=2)))
    pos_new = pos + d_fov
    pos_new[:, 0] = np.clip(pos_new[:, 0], 0, nr - 1)
    pos_new[:, 1] = np.clip(pos_new[:, 1], 0, nc - 1)
    mask[pos_new[:, 0], pos_new[:, 1]] = 1
    return mask


def get_distances(positions: np.array):
    """given a positions vector, return distance matrix between all the positions.
    kindly provided by stack overflow
    https://stackoverflow.com/questions/71289996/what-is-the-best-way-to-get-objects-distances-to-one-another-using-numpy/71290055#71290055"""
    relative_positions = positions[None, :, :] - positions[:, None, :]
    distances = np.abs(relative_positions).sum(axis=2)
    return distances


def get_comm_dist_mat(ranges):
    relative_positions = ranges[None, :, :] + ranges[:, None, :]
    comm_ranges = np.abs(relative_positions).sum(axis=2)
    return comm_ranges
