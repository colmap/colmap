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
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import pytest

import pycolmap

from .utils import (
    Metrics,
    compute_abs_errors,
    compute_auc,
    compute_avg_metrics,
    compute_recall,
    compute_rel_errors,
    diff_metrics,
    get_scores,
    normalize_vec,
    vec_angular_dist_deg,
)


class TestNormalizeVec:
    def test_unit_vector(self):
        vec = np.array([1.0, 0.0, 0.0])
        normalized = normalize_vec(vec)
        np.testing.assert_allclose(normalized, vec)
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)

    def test_non_unit_vector(self):
        vec = np.array([3.0, 4.0, 0.0])
        normalized = normalize_vec(vec)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_allclose(normalized, expected)
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)

    def test_zero_vector(self):
        vec = np.array([0.0, 0.0, 0.0])
        normalized = normalize_vec(vec)
        assert np.linalg.norm(normalized) < 1e-8


class TestVecAngularDistDeg:
    def test_identical_vectors(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_opposite_vectors(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        np.testing.assert_almost_equal(dist, 180.0)

    def test_perpendicular_vectors(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        np.testing.assert_almost_equal(dist, 90.0)

    def test_45_degree_vectors(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 1.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        np.testing.assert_almost_equal(dist, 45.0)

    def test_non_unit_vectors(self):
        vec1 = np.array([2.0, 0.0, 0.0])
        vec2 = np.array([0.0, 3.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        # Should normalize internally
        np.testing.assert_almost_equal(dist, 90.0)

    def test_clipping_behavior(self):
        # Test that dot product is clipped to [-1, 1] to avoid numerical issues
        vec1 = np.array([1.0, 1e-10, 1e-10])
        vec2 = np.array([1.0, -1e-10, -1e-10])
        dist = vec_angular_dist_deg(vec1, vec2)
        # Should not raise error even with potential numerical issues
        assert 0 <= dist <= 180


class TestComputeAuc:
    def test_simple_uniform_errors(self):
        errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        thresholds = np.array([0.25, 0.5, 1.0])
        aucs = compute_auc(errors, thresholds)
        np.testing.assert_almost_equal(aucs[0], 24.0, decimal=5)
        np.testing.assert_almost_equal(aucs[1], 50.0, decimal=5)
        np.testing.assert_almost_equal(aucs[2], 75.0, decimal=5)

    def test_all_errors_zero(self):
        errors = np.array([0.0, 0.0, 0.0])
        thresholds = np.array([0.5, 1.0])
        aucs = compute_auc(errors, thresholds)
        np.testing.assert_array_almost_equal(aucs, [100.0, 100.0])

    def test_empty_errors(self):
        errors = np.array([])
        thresholds = np.array([0.5, 1.0])
        with pytest.raises(ValueError, match="No errors to evaluate"):
            compute_auc(errors, thresholds)

    def test_all_errors_above_threshold(self):
        errors = np.array([10.0, 20.0, 30.0])
        thresholds = np.array([5.0])
        aucs = compute_auc(errors, thresholds)
        np.testing.assert_almost_equal(aucs[0], 0.0)

    def test_all_errors_below_threshold(self):
        errors = np.array([0.1, 0.2, 0.3])
        thresholds = np.array([1.0])
        aucs = compute_auc(errors, thresholds)
        np.testing.assert_almost_equal(aucs[0], 85.0, decimal=5)

    def test_inf_errors(self):
        errors = np.array([0.1, 0.2, np.inf, np.inf])
        thresholds = np.array([0.5, 1.0])
        aucs = compute_auc(errors, thresholds)
        assert np.all(aucs >= 0)
        assert np.all(aucs <= 100)

    def test_single_error(self):
        errors = np.array([0.5])
        thresholds = np.array([0.3, 1.0])
        aucs = compute_auc(errors, thresholds)
        assert len(aucs) == 2
        np.testing.assert_almost_equal(aucs[0], 0.0)
        np.testing.assert_almost_equal(aucs[1], 75.0)


class TestComputeRecall:
    def test_basic_recall(self):
        errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        thresholds = np.array([0.05, 0.25, 0.5, 1.0])
        recalls = compute_recall(errors, thresholds)
        assert len(recalls) == 4
        assert recalls[3] >= recalls[2] >= recalls[1] >= recalls[0]
        assert np.all(recalls >= 0)
        assert np.all(recalls <= 100)

    def test_empty_errors(self):
        errors = np.array([])
        thresholds = np.array([0.5, 1.0])
        with pytest.raises(ValueError, match="No errors to evaluate"):
            compute_recall(errors, thresholds)

    def test_all_errors_above_threshold(self):
        errors = np.array([10.0, 20.0, 30.0])
        thresholds = np.array([5.0])
        recalls = compute_recall(errors, thresholds)
        np.testing.assert_almost_equal(recalls[0], 0.0)

    def test_all_errors_below_threshold(self):
        errors = np.array([0.1, 0.2, 0.3])
        thresholds = np.array([1.0])
        recalls = compute_recall(errors, thresholds)
        np.testing.assert_almost_equal(recalls[0], 100.0)

    def test_exact_threshold(self):
        errors = np.array([0.1, 0.5, 0.9])
        thresholds = np.array([0.5])
        recalls = compute_recall(errors, thresholds)
        np.testing.assert_almost_equal(recalls[0], 200.0 / 3.0)

    def test_multiple_thresholds(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        thresholds = np.array([2.0, 3.0, 4.0])
        recalls = compute_recall(errors, thresholds)
        np.testing.assert_almost_equal(recalls[0], 40.0)
        np.testing.assert_almost_equal(recalls[1], 60.0)
        np.testing.assert_almost_equal(recalls[2], 80.0)


class TestComputeAvgMetrics:
    def test_single_scene(self):
        scene_metrics = {
            "scene1": Metrics(
                aucs=np.array([10.0, 20.0, 30.0]),
                recalls=np.array([15.0, 25.0, 35.0]),
                error_thresholds=np.array([0.5, 1.0, 2.0]),
                error_type="relative_auc",
                num_images=100,
                num_reg_images=90,
                num_components=1,
                largest_component=90,
            )
        }
        aucs, recalls = compute_avg_metrics(scene_metrics)
        np.testing.assert_array_equal(aucs, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(recalls, [15.0, 25.0, 35.0])

    def test_multiple_scenes(self):
        scene_metrics = {
            "scene1": Metrics(
                aucs=np.array([10.0, 20.0, 30.0]),
                recalls=np.array([15.0, 25.0, 35.0]),
                error_thresholds=np.array([0.5, 1.0, 2.0]),
                error_type="relative_auc",
                num_images=100,
                num_reg_images=90,
                num_components=1,
                largest_component=90,
            ),
            "scene2": Metrics(
                aucs=np.array([20.0, 30.0, 40.0]),
                recalls=np.array([25.0, 35.0, 45.0]),
                error_thresholds=np.array([0.5, 1.0, 2.0]),
                error_type="relative_auc",
                num_images=100,
                num_reg_images=90,
                num_components=1,
                largest_component=90,
            ),
        }
        aucs, recalls = compute_avg_metrics(scene_metrics)
        np.testing.assert_array_equal(aucs, [15.0, 25.0, 35.0])
        np.testing.assert_array_equal(recalls, [20.0, 30.0, 40.0])

    def test_skip_special_scenes(self):
        scene_metrics = {
            "scene1": Metrics(
                aucs=np.array([10.0, 20.0, 30.0]),
                recalls=np.array([15.0, 25.0, 35.0]),
                error_thresholds=np.array([0.5, 1.0, 2.0]),
                error_type="relative_auc",
                num_images=100,
                num_reg_images=90,
                num_components=1,
                largest_component=90,
            ),
            "__avg__": Metrics(
                aucs=np.array([50.0, 60.0, 70.0]),
                recalls=np.array([55.0, 65.0, 75.0]),
                error_thresholds=np.array([0.5, 1.0, 2.0]),
                error_type="relative_auc",
                num_images=100,
                num_reg_images=90,
                num_components=1,
                largest_component=90,
            ),
            "__all__": Metrics(
                aucs=np.array([80.0, 90.0, 100.0]),
                recalls=np.array([85.0, 95.0, 105.0]),
                error_thresholds=np.array([0.5, 1.0, 2.0]),
                error_type="relative_auc",
                num_images=100,
                num_reg_images=90,
                num_components=1,
                largest_component=90,
            ),
        }
        aucs, recalls = compute_avg_metrics(scene_metrics)
        # Should only average scene1, not __avg__ or __all__
        np.testing.assert_array_equal(aucs, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(recalls, [15.0, 25.0, 35.0])


class TestGetScores:
    def test_get_auc_scores(self):
        metrics = Metrics(
            aucs=np.array([10.0, 20.0, 30.0]),
            recalls=np.array([15.0, 25.0, 35.0]),
            error_thresholds=np.array([0.5, 1.0, 2.0]),
            error_type="relative_auc",
            num_images=100,
            num_reg_images=90,
            num_components=1,
            largest_component=90,
        )
        scores = get_scores("relative_auc", metrics)
        np.testing.assert_array_equal(scores, metrics.aucs)

    def test_get_recall_scores(self):
        metrics = Metrics(
            aucs=np.array([10.0, 20.0, 30.0]),
            recalls=np.array([15.0, 25.0, 35.0]),
            error_thresholds=np.array([0.5, 1.0, 2.0]),
            error_type="relative_recall",
            num_images=100,
            num_reg_images=90,
            num_components=1,
            largest_component=90,
        )
        scores = get_scores("relative_recall", metrics)
        np.testing.assert_array_equal(scores, metrics.recalls)


class TestDiffMetrics:
    def test_nominal(self):
        metrics_a = {
            "dataset1": {
                "category1": {
                    "scene1": Metrics(
                        aucs=np.array([20.0, 30.0, 40.0]),
                        recalls=np.array([25.0, 35.0, 45.0]),
                        error_thresholds=np.array([0.5, 1.0, 2.0]),
                        error_type="relative_auc",
                        num_images=100,
                        num_reg_images=90,
                        num_components=2,
                        largest_component=80,
                    )
                }
            }
        }
        metrics_b = {
            "dataset1": {
                "category1": {
                    "scene1": Metrics(
                        aucs=np.array([10.0, 20.0, 30.0]),
                        recalls=np.array([15.0, 25.0, 35.0]),
                        error_thresholds=np.array([0.5, 1.0, 2.0]),
                        error_type="relative_auc",
                        num_images=100,
                        num_reg_images=85,
                        num_components=1,
                        largest_component=85,
                    )
                }
            }
        }
        diff = diff_metrics(metrics_a, metrics_b)

        scene_diff = diff["dataset1"]["category1"]["scene1"]
        np.testing.assert_array_equal(scene_diff.aucs, [10.0, 10.0, 10.0])
        np.testing.assert_array_equal(scene_diff.recalls, [10.0, 10.0, 10.0])
        assert scene_diff.num_reg_images == 5
        assert scene_diff.num_components == 1


def create_test_reconstruction():
    pycolmap.set_random_seed(0)
    synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
    synthetic_dataset_options.num_cameras_per_rig = 1
    synthetic_dataset_options.num_frames_per_rig = 5
    synthetic_dataset_options.num_points3D = 0
    return pycolmap.synthesize_dataset(synthetic_dataset_options)


class TestComputeAbsErrors:
    def test_identical_reconstruction(self):
        reconstruction = create_test_reconstruction()

        dts, dRs = compute_abs_errors(
            sparse_gt=reconstruction, sparse=reconstruction
        )

        assert len(dts) == reconstruction.num_images()
        assert len(dRs) == reconstruction.num_images()
        np.testing.assert_allclose(dts, 0.0, atol=1e-10)
        np.testing.assert_allclose(dRs, 0.0, atol=1e-10)

    def test_transformed_reconstruction(self):
        gt_reconstruction = create_test_reconstruction()
        reconstruction = create_test_reconstruction()
        translation = np.array([1, 2, 3])
        for frame in reconstruction.frames.values():
            world_from_rig = frame.rig_from_world.inverse()
            world_from_rig.rotation = (
                world_from_rig.rotation * pycolmap.Rotation3d([0, 1, 0, 0])
            )
            world_from_rig.translation += translation
            frame.rig_from_world = world_from_rig.inverse()

        dts, dRs = compute_abs_errors(
            sparse_gt=gt_reconstruction, sparse=reconstruction
        )

        assert len(dts) == reconstruction.num_images()
        assert len(dRs) == reconstruction.num_images()
        np.testing.assert_allclose(dts, np.linalg.norm(translation), atol=1e-10)
        np.testing.assert_allclose(dRs, 180.0, atol=1e-10)


class TestComputeRelErrors:
    def test_identical_reconstruction(self):
        reconstruction = create_test_reconstruction()

        dts, dRs = compute_rel_errors(
            sparse_gt=reconstruction,
            sparse=reconstruction,
            min_proj_center_dist=0.01,
        )

        num_images = reconstruction.num_images()
        expected_num_errors = num_images * (num_images - 1)
        assert len(dts) == expected_num_errors
        assert len(dRs) == expected_num_errors
        np.testing.assert_allclose(dts, 0.0, atol=1e-5)
        np.testing.assert_allclose(dRs, 0.0, atol=1e-5)

    def test_transformed_reconstruction(self):
        gt_reconstruction = create_test_reconstruction()
        reconstruction = create_test_reconstruction()
        reconstruction.transform(
            pycolmap.Sim3d(
                1.0,
                pycolmap.Rotation3d(np.array([0, 1, 0, 0])),
                np.array([1, 2, 3]),
            )
        )

        dts, dRs = compute_rel_errors(
            sparse_gt=gt_reconstruction,
            sparse=reconstruction,
            min_proj_center_dist=0.01,
        )

        num_images = reconstruction.num_images()
        expected_num_errors = num_images * (num_images - 1)
        assert len(dts) == expected_num_errors
        assert len(dRs) == expected_num_errors
        np.testing.assert_allclose(dts, 0.0, atol=1e-5)
        np.testing.assert_allclose(dRs, 0.0, atol=1e-5)

    def test_different_reconstructions(self):
        gt_reconstruction = create_test_reconstruction()
        reconstruction = create_test_reconstruction()
        for image in reconstruction.images.values():
            image.frame.rig_from_world.rotation = (
                pycolmap.Rotation3d(np.array([0, 1, 0, 0]))
                * image.frame.rig_from_world.rotation
            )
            image.frame.rig_from_world.translation += np.array([1, 2, 3])

        dts, dRs = compute_rel_errors(
            sparse_gt=gt_reconstruction,
            sparse=reconstruction,
            min_proj_center_dist=0.01,
        )

        num_images = reconstruction.num_images()
        expected_num_errors = num_images * (num_images - 1)
        assert len(dts) == expected_num_errors
        assert len(dRs) == expected_num_errors
        assert np.all(dts > 0.1)
        assert np.all(dRs > 0.1)
