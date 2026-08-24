# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np

from benchmark.reconstruction.evaluation.utils import compute_auc
from benchmark.reconstruction.synthetic_pose_noise import (
    _ordered_pair_indices,
    aucs_from_errors,
    axis_angle_to_rotation_matrices,
)


def test_axis_angle_to_rotation_matrices():
    rotations = axis_angle_to_rotation_matrices(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]])
    )

    np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-15)
    np.testing.assert_allclose(
        rotations[1],
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-15,
    )


def test_aucs_from_errors_matches_benchmark():
    errors = np.array([0.0, 0.0005, 0.2, 0.7, 2.0, 20.0])
    thresholds = np.array([0.5, 1.0, 5.0, 10.0])

    expected = compute_auc(errors, thresholds, min_error=0.001)
    actual = aucs_from_errors(errors, thresholds, min_error=0.001)

    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_ordered_pair_indices_respect_gt_components_and_outliers():
    src, tgt = _ordered_pair_indices(np.array([0, 0, 1, 1, -1]))

    assert set(zip(src, tgt, strict=True)) == {
        (0, 1),
        (1, 0),
        (2, 3),
        (3, 2),
    }
