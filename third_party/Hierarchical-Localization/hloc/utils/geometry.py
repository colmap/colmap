import numpy as np
import pycolmap


def to_homogeneous(p):
    return np.pad(p, ((0, 0),) * (p.ndim - 1) + ((0, 1),), constant_values=1)


def compute_epipolar_errors(j_from_i: pycolmap.Rigid3d, p2d_i, p2d_j):
    j_E_i = pycolmap.essential_matrix_from_pose(j_from_i)
    l2d_j = to_homogeneous(p2d_i) @ j_E_i.T
    l2d_i = to_homogeneous(p2d_j) @ j_E_i
    dist = np.abs(np.sum(to_homogeneous(p2d_i) * l2d_i, axis=1))
    errors_i = dist / np.linalg.norm(l2d_i[:, :2], axis=1)
    errors_j = dist / np.linalg.norm(l2d_j[:, :2], axis=1)
    return errors_i, errors_j
