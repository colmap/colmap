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

from .statistics import (
    NoiseCeiling,
    ceiling_gap_interval,
    classify_ceiling_gap,
    classify_difference,
    load_noise_ceiling,
    paired_difference_interval,
    save_noise_ceiling,
)


def test_paired_difference_constant_effect():
    scores_b = np.arange(6 * 4 * 2, dtype=float).reshape(6, 4, 2)
    scores_a = scores_b + np.array([0.5, 1.0])

    result = paired_difference_interval(
        scores_a, scores_b, num_bootstrap_samples=100, random_seed=7
    )

    np.testing.assert_allclose(result.estimate, [0.5, 1.0])
    np.testing.assert_allclose(result.lower, result.estimate)
    np.testing.assert_allclose(result.upper, result.estimate)
    assert classify_difference(result.lower[0], result.upper[0], 0.25) == (
        "A better"
    )


def test_paired_difference_is_reproducible_and_simultaneous():
    rng = np.random.default_rng(4)
    scores_b = rng.normal(size=(12, 5, 4))
    scores_a = scores_b + rng.normal(0.2, 0.5, size=scores_b.shape)

    result1 = paired_difference_interval(
        scores_a, scores_b, num_bootstrap_samples=500, random_seed=9
    )
    result2 = paired_difference_interval(
        scores_a, scores_b, num_bootstrap_samples=500, random_seed=9
    )

    np.testing.assert_array_equal(result1.lower, result2.lower)
    np.testing.assert_array_equal(result1.upper, result2.upper)
    assert np.all(result1.lower <= result1.estimate)
    assert np.all(result1.estimate <= result1.upper)
    assert np.all(result1.adjusted_p_values >= 0)
    assert np.all(result1.adjusted_p_values <= 1)


def test_ceiling_gap_constant_effect():
    run = np.full((5, 3, 2), 95.0)
    ceiling = np.full((5, 20, 2), 95.3)

    result = ceiling_gap_interval(
        run, ceiling, num_bootstrap_samples=100, random_seed=3
    )

    np.testing.assert_allclose(result.estimate, [0.3, 0.3])
    assert classify_ceiling_gap(result.lower[0], result.upper[0], 0.5) == (
        "ceiling-limited"
    )
    assert classify_ceiling_gap(0.7, 0.9, 0.5) == "headroom remains"
    assert classify_ceiling_gap(0.3, 0.7, 0.5) == "inconclusive"


def test_noise_ceiling_round_trip(tmp_path):
    path = tmp_path / "ceiling.npz"
    expected = NoiseCeiling(
        scene_keys=("dataset/category/a", "dataset/category/b"),
        scores=np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
        thresholds=np.array([0.5, 1.0, 5.0, 10.0]),
        error_type="relative_auc",
        metadata={"calibrated": False, "model": "test"},
    )

    save_noise_ceiling(path, expected)
    actual = load_noise_ceiling(path)

    assert actual.scene_keys == expected.scene_keys
    np.testing.assert_array_equal(actual.scores, expected.scores)
    np.testing.assert_array_equal(actual.thresholds, expected.thresholds)
    assert actual.error_type == expected.error_type
    assert actual.metadata == expected.metadata
