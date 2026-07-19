import numpy as np

import pycolmap


def test_essential_matrix_from_pose():
    rotation = pycolmap.Rotation3d(axis_angle=np.array([0.0, 0.1, 0.0]))
    translation = np.array([1.0, 0.0, 0.0])
    rigid = pycolmap.Rigid3d(rotation=rotation, translation=translation)
    essential_matrix = pycolmap.essential_matrix_from_pose(rigid)
    assert essential_matrix.shape == (3, 3)
