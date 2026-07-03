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

import argparse
from pathlib import Path

import numpy as np
import pytest

import pycolmap

from .utils import (
    OUTLIER_RECON_ID,
    Metrics,
    SceneInfo,
    _parse_gpu_index,
    aggregate_scene_metrics,
    compute_abs_errors,
    compute_auc,
    compute_avg_metrics,
    compute_grouped_abs_errors,
    compute_grouped_rel_errors,
    compute_recall,
    diff_metrics,
    filter_smallest_scenes_per_category,
    get_scores,
)


def _make_scene_info(category: str, scene: str, num_images: int) -> SceneInfo:
    return SceneInfo(
        dataset="dummy",
        category=category,
        scene=scene,
        num_images=num_images,
        workspace_path=Path("/tmp/workspace"),
        image_path=Path("/tmp/images"),
        sparse_gt_path=Path("/tmp/sparse_gt"),
        has_camera_priors=False,
        colmap_extra_args=[],
    )


class TestFilterSmallestScenesPerCategory:
    def test_picks_smallest_per_category(self):
        scenes = [
            _make_scene_info("a", "a3", 30),
            _make_scene_info("a", "a1", 10),
            _make_scene_info("a", "a2", 20),
            _make_scene_info("b", "b2", 5),
            _make_scene_info("b", "b1", 1),
        ]
        result = filter_smallest_scenes_per_category(scenes, num_scenes=2)
        names = [(s.category, s.scene) for s in result]
        assert names == [("a", "a1"), ("a", "a2"), ("b", "b2"), ("b", "b1")]

    def test_preserves_input_order(self):
        scenes = [
            _make_scene_info("a", "a3", 30),
            _make_scene_info("a", "a1", 10),
            _make_scene_info("a", "a2", 20),
        ]
        result = filter_smallest_scenes_per_category(scenes, num_scenes=2)
        # Smallest are a1 and a2, but the original order (a3, a1, a2) must
        # be preserved among the kept scenes.
        assert [s.scene for s in result] == ["a1", "a2"]

    def test_num_scenes_larger_than_category_size(self):
        scenes = [
            _make_scene_info("a", "a1", 10),
            _make_scene_info("a", "a2", 20),
            _make_scene_info("b", "b1", 5),
        ]
        result = filter_smallest_scenes_per_category(scenes, num_scenes=10)
        # All scenes are kept since each category has fewer than num_scenes.
        assert [s.scene for s in result] == ["a1", "a2", "b1"]

    def test_num_scenes_one(self):
        scenes = [
            _make_scene_info("a", "a1", 10),
            _make_scene_info("a", "a2", 5),
            _make_scene_info("b", "b1", 100),
            _make_scene_info("b", "b2", 50),
        ]
        result = filter_smallest_scenes_per_category(scenes, num_scenes=1)
        assert sorted((s.category, s.scene) for s in result) == [
            ("a", "a2"),
            ("b", "b2"),
        ]

    def test_empty_input(self):
        assert filter_smallest_scenes_per_category([], num_scenes=3) == []

    def test_ties_broken_stably(self):
        # When several scenes share the same num_images, sorting must be
        # stable so we keep the ones that appeared first in the input.
        scenes = [
            _make_scene_info("a", "a1", 10),
            _make_scene_info("a", "a2", 10),
            _make_scene_info("a", "a3", 10),
        ]
        result = filter_smallest_scenes_per_category(scenes, num_scenes=2)
        assert [s.scene for s in result] == ["a1", "a2"]


class TestParseGpuIndex:
    @staticmethod
    def _make_args(gpu_index: str) -> argparse.Namespace:
        return argparse.Namespace(gpu_index=gpu_index)

    def test_single_gpu(self):
        assert _parse_gpu_index(self._make_args("0")) == [0]

    def test_multiple_gpus(self):
        assert _parse_gpu_index(self._make_args("0,1,2")) == [0, 1, 2]

    def test_trailing_comma(self):
        assert _parse_gpu_index(self._make_args("1,")) == [1]

    def test_empty_string(self):
        assert _parse_gpu_index(self._make_args("")) == [-1]

    def test_only_commas(self):
        assert _parse_gpu_index(self._make_args(",")) == [-1]

    def test_auto_detect(self, monkeypatch):
        monkeypatch.setattr(pycolmap, "has_cuda", True)
        monkeypatch.setattr(
            pycolmap, "get_num_cuda_devices", lambda: 3, raising=False
        )
        assert _parse_gpu_index(self._make_args("-1")) == [0, 1, 2]

    def test_auto_detect_no_devices(self, monkeypatch):
        monkeypatch.setattr(pycolmap, "has_cuda", True)
        monkeypatch.setattr(
            pycolmap, "get_num_cuda_devices", lambda: 0, raising=False
        )
        assert _parse_gpu_index(self._make_args("-1")) == [-1]


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


class TestAggregateSceneMetrics:
    @staticmethod
    def _make_metrics(aucs, recalls, errors, num_images=100, num_reg_images=90):
        return Metrics(
            aucs=np.array(aucs, dtype=float),
            recalls=np.array(recalls, dtype=float),
            error_thresholds=np.array([0.5, 1.0, 2.0]),
            error_type="relative_auc",
            num_images=num_images,
            num_reg_images=num_reg_images,
            num_components=1,
            largest_component=num_reg_images,
            errors=np.array(errors, dtype=float),
            position_accuracy_gt=0.01,
        )

    def test_avg_and_all(self):
        scene_metrics = [
            (
                "scene1",
                self._make_metrics([10, 20, 30], [15, 25, 35], [0.1, 0.5]),
            ),
            (
                "scene2",
                self._make_metrics([20, 30, 40], [25, 35, 45], [0.2, 1.5]),
            ),
        ]
        summary = aggregate_scene_metrics(
            scene_metrics,
            error_thresholds=np.array([0.5, 1.0, 2.0]),
            error_type="relative_auc",
        )
        np.testing.assert_array_equal(
            summary["__avg__"].aucs, [15.0, 25.0, 35.0]
        )
        np.testing.assert_array_equal(
            summary["__avg__"].recalls, [20.0, 30.0, 40.0]
        )
        assert summary["__avg__"].num_images == 100
        assert summary["__avg__"].num_reg_images == 90
        np.testing.assert_array_equal(
            summary["__all__"].errors, [0.1, 0.5, 0.2, 1.5]
        )
        # __all__ aggregates totals (not means).
        assert summary["__all__"].num_images == 200
        assert summary["__all__"].num_reg_images == 180

    def test_skips_special_entries(self):
        real = self._make_metrics([10, 20, 30], [15, 25, 35], [0.1])
        special = self._make_metrics(
            [99, 99, 99], [99, 99, 99], [9.0], num_images=999
        )
        summary = aggregate_scene_metrics(
            [("scene1", real), ("__avg__", special), ("__all__", special)],
            error_thresholds=np.array([0.5, 1.0, 2.0]),
            error_type="relative_auc",
        )
        np.testing.assert_array_equal(summary["__avg__"].aucs, real.aucs)
        np.testing.assert_array_equal(summary["__all__"].errors, [0.1])

    def test_empty_input(self):
        assert (
            aggregate_scene_metrics(
                [],
                error_thresholds=np.array([0.5, 1.0, 2.0]),
                error_type="relative_auc",
            )
            == {}
        )

    def test_no_errors_omits_all(self):
        # When no scene carries raw errors (e.g. metrics restored without
        # the errors field), __all__ cannot be reconstructed.
        scene_metrics = [
            ("scene1", self._make_metrics([10, 20, 30], [15, 25, 35], [])),
            ("scene2", self._make_metrics([20, 30, 40], [25, 35, 45], [])),
        ]
        summary = aggregate_scene_metrics(
            scene_metrics,
            error_thresholds=np.array([0.5, 1.0, 2.0]),
            error_type="relative_auc",
        )
        assert "__all__" not in summary
        assert "__avg__" in summary


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


def sub_reconstruction(reconstruction, keep_names):
    """Build a copy of reconstruction containing only the given image names.

    Mirrors what a real sub-model loaded from disk looks like: its images map
    holds exactly the registered subset. Rebuilding from a private copy keeps
    the source reconstruction untouched.
    """
    keep_names = set(keep_names)
    source = pycolmap.Reconstruction(reconstruction)
    keep_frame_ids = {
        image.frame_id
        for image in source.images.values()
        if image.name in keep_names
    }
    sub = pycolmap.Reconstruction()
    for camera in source.cameras.values():
        sub.add_camera(camera)
    for rig in source.rigs.values():
        sub.add_rig(rig)
    for frame in source.frames.values():
        if frame.frame_id in keep_frame_ids:
            frame.reset_rig_ptr()
            sub.add_frame(frame)
    for image in source.images.values():
        if image.name in keep_names:
            image.reset_camera_ptr()
            image.reset_frame_ptr()
            sub.add_image(image)
    for frame_id in keep_frame_ids:
        sub.register_frame(frame_id)
    return sub


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


def _single_gt_cluster(reconstruction) -> dict[str, int]:
    """Map every image of a reconstruction to a single GT cluster (id 0)."""
    return {image.name: 0 for image in reconstruction.images.values()}


class TestComputeGroupedRelErrors:
    def test_identical_reconstruction(self):
        reconstruction = create_test_reconstruction()

        errors = compute_grouped_rel_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=_single_gt_cluster(reconstruction),
            min_proj_center_dist=0.01,
        )

        num_images = reconstruction.num_images()
        # A n B covers every ordered pair; A - B and B - A are empty.
        assert len(errors) == num_images * (num_images - 1)
        np.testing.assert_allclose(errors, 0.0, atol=1e-5)

    def test_transformed_reconstruction(self):
        gt_reconstruction = create_test_reconstruction()
        reconstruction = create_test_reconstruction()
        # A global similarity transform leaves relative poses unchanged.
        reconstruction.transform(
            pycolmap.Sim3d(
                1.0,
                pycolmap.Rotation3d(np.array([0, 1, 0, 0])),
                np.array([1, 2, 3]),
            )
        )

        errors = compute_grouped_rel_errors(
            sparse_gt=gt_reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=_single_gt_cluster(gt_reconstruction),
            min_proj_center_dist=0.01,
        )

        num_images = reconstruction.num_images()
        assert len(errors) == num_images * (num_images - 1)
        np.testing.assert_allclose(errors, 0.0, atol=1e-5)

    def test_different_reconstructions(self):
        gt_reconstruction = create_test_reconstruction()
        reconstruction = create_test_reconstruction()
        for image in reconstruction.images.values():
            image.frame.rig_from_world.rotation = (
                pycolmap.Rotation3d(np.array([0, 1, 0, 0]))
                * image.frame.rig_from_world.rotation
            )
            image.frame.rig_from_world.translation += np.array([1, 2, 3])

        errors = compute_grouped_rel_errors(
            sparse_gt=gt_reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=_single_gt_cluster(gt_reconstruction),
            min_proj_center_dist=0.01,
        )

        num_images = reconstruction.num_images()
        assert len(errors) == num_images * (num_images - 1)
        assert np.all(errors > 0.1)

    def test_nothing_registered_maxes_all_gt_edges(self):
        # No estimated sub-models: set A is empty, so every GT edge is in
        # B - A and must be scored as the maximum (180 degrees).
        reconstruction = create_test_reconstruction()

        errors = compute_grouped_rel_errors(
            sparse_gt=reconstruction,
            sub_models=[],
            image_name_to_gt_recon_ids=_single_gt_cluster(reconstruction),
            min_proj_center_dist=0.01,
        )

        num_images = reconstruction.num_images()
        assert len(errors) == num_images * (num_images - 1)
        np.testing.assert_allclose(errors, 180.0)

    def test_merged_estimate_of_separate_gt_clusters_maxes_cross_edges(self):
        # Two GT clusters, each perfectly reconstructed in its own sub-model.
        # There are no cross-cluster edges in B, and the within-cluster edges
        # are all in A n B with ~0 error.
        reconstruction = create_test_reconstruction()
        names = sorted(image.name for image in reconstruction.images.values())
        half = len(names) // 2
        image_name_to_gt_recon_ids = {
            name: (0 if i < half else 1) for i, name in enumerate(names)
        }

        errors = compute_grouped_rel_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
            min_proj_center_dist=0.01,
        )

        # Edges within a cluster: A n B (scored). Cross-cluster edges are in
        # A - B (the single sub-model connects everything) and set to 180.
        n0 = half
        n1 = len(names) - half
        num_within = n0 * (n0 - 1) + n1 * (n1 - 1)
        num_cross = len(names) * (len(names) - 1) - num_within
        assert len(errors) == len(names) * (len(names) - 1)
        np.testing.assert_allclose(np.sort(errors)[:num_within], 0.0, atol=1e-5)
        assert int(np.sum(np.isclose(errors, 180.0))) == num_cross

    def test_fragmented_estimate_of_merged_gt_maxes_cross_edges(self):
        # One GT cluster (merged reconstruction) that the estimate splits into
        # two disjoint sub-models (fragmented estimation). Within-fragment edges
        # are in A n B (scored ~0); the GT edges bridging the two fragments are
        # in B - A and set to 180.
        reconstruction = create_test_reconstruction()
        names = sorted(image.name for image in reconstruction.images.values())
        half = len(names) // 2
        sub_model_0 = sub_reconstruction(reconstruction, names[:half])
        sub_model_1 = sub_reconstruction(reconstruction, names[half:])
        image_name_to_gt_recon_ids = {name: 0 for name in names}

        errors = compute_grouped_rel_errors(
            sparse_gt=reconstruction,
            sub_models=[sub_model_0, sub_model_1],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
            min_proj_center_dist=0.01,
        )

        n0 = half
        n1 = len(names) - half
        num_within = n0 * (n0 - 1) + n1 * (n1 - 1)
        num_cross = len(names) * (len(names) - 1) - num_within
        assert len(errors) == len(names) * (len(names) - 1)
        np.testing.assert_allclose(np.sort(errors)[:num_within], 0.0, atol=1e-5)
        assert int(np.sum(np.isclose(errors, 180.0))) == num_cross

    def test_mismatched_gt_and_estimate_cluster_boundaries(self):
        # GT splits the images 1/3 vs 2/3, but the estimate splits them 2/3 vs
        # 1/3, so the two cluster boundaries disagree. Ordered pairs that share
        # both an estimated sub-model and a GT cluster are scored (~0 for a
        # perfect estimate); pairs grouped by only one of the two are maxed.
        reconstruction = create_test_reconstruction()
        names = sorted(image.name for image in reconstruction.images.values())
        n = len(names)
        third = n // 3
        two_thirds = 2 * n // 3
        # GT: first third -> cluster 0, remaining two thirds -> cluster 1.
        image_name_to_gt_recon_ids = {
            name: (0 if i < third else 1) for i, name in enumerate(names)
        }
        # Estimate: first two thirds and remaining third form two sub-models.
        sub_model_0 = sub_reconstruction(reconstruction, names[:two_thirds])
        sub_model_1 = sub_reconstruction(reconstruction, names[two_thirds:])

        errors = compute_grouped_rel_errors(
            sparse_gt=reconstruction,
            sub_models=[sub_model_0, sub_model_1],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
            min_proj_center_dist=0.01,
        )

        # Three groups by (sub-model, GT cluster): P = names[:third] (sub 0,
        # cluster 0), Q = names[third:two_thirds] (sub 0, cluster 1), R =
        # names[two_thirds:] (sub 1, cluster 1). Only intra-group edges share
        # both a sub-model and a cluster (A n B, ~0). P-Q share a sub-model but
        # not a cluster (A - B), and Q-R share a cluster but not a sub-model
        # (B - A); both are maxed to 180.
        n_p = third
        n_q = two_thirds - third
        n_r = n - two_thirds
        num_scored = n_p * (n_p - 1) + n_q * (n_q - 1) + n_r * (n_r - 1)
        num_maxed = 2 * (n_p * n_q) + 2 * (n_q * n_r)
        assert len(errors) == num_scored + num_maxed
        np.testing.assert_allclose(np.sort(errors)[:num_scored], 0.0, atol=1e-5)
        assert int(np.sum(np.isclose(errors, 180.0))) == num_maxed

    def test_registered_outlier_edges_are_maxed(self):
        # A single sub-model connects an outlier to every real image. The
        # outlier is never in a GT reconstruction, so all edges touching it are
        # in A - B and set to 180; the remaining real edges are A n B (~0).
        reconstruction = create_test_reconstruction()
        names = sorted(image.name for image in reconstruction.images.values())
        outlier = names[0]
        image_name_to_gt_recon_ids = {
            name: (OUTLIER_RECON_ID if name == outlier else 0)
            for name in names
        }

        errors = compute_grouped_rel_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
            min_proj_center_dist=0.01,
        )

        num_images = reconstruction.num_images()
        assert len(errors) == num_images * (num_images - 1)
        # The 2 * (num_images - 1) ordered edges touching the outlier are maxed.
        num_maxed = int(np.sum(np.isclose(errors, 180.0)))
        assert num_maxed == 2 * (num_images - 1)

    def test_outlier_present_in_gt_maxes_its_edges(self):
        # An image that exists in the GT reconstruction and is perfectly
        # registered, but is flagged as an outlier, must still have large
        # edges: every edge touching it is in A - B and set to 180, while the
        # edges among the real images stay in A n B (~0).
        reconstruction = create_test_reconstruction()
        gt_names = [image.name for image in reconstruction.images.values()]
        outlier = sorted(gt_names)[0]
        assert outlier in gt_names  # The outlier is present in the GT.
        image_name_to_gt_recon_ids = {
            name: (OUTLIER_RECON_ID if name == outlier else 0)
            for name in gt_names
        }

        errors = compute_grouped_rel_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
            min_proj_center_dist=0.01,
        )

        n = reconstruction.num_images()
        assert len(errors) == n * (n - 1)
        # All 2 * (n - 1) ordered edges touching the outlier are maxed; the
        # (n - 1) * (n - 2) edges among the real images are ~0.
        assert int(np.sum(np.isclose(errors, 180.0))) == 2 * (n - 1)
        np.testing.assert_allclose(
            np.sort(errors)[: (n - 1) * (n - 2)], 0.0, atol=1e-5
        )

    def test_outliers_excluded_from_gt_edges(self):
        # Nothing is registered, so every scored pair comes from B - A. The
        # outlier forms no GT edges, so it contributes none of them.
        reconstruction = create_test_reconstruction()
        names = sorted(image.name for image in reconstruction.images.values())
        outlier = names[0]
        image_name_to_gt_recon_ids = {
            name: (OUTLIER_RECON_ID if name == outlier else 0)
            for name in names
        }

        errors = compute_grouped_rel_errors(
            sparse_gt=reconstruction,
            sub_models=[],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
            min_proj_center_dist=0.01,
        )

        num_real = len(names) - 1
        assert len(errors) == num_real * (num_real - 1)
        np.testing.assert_allclose(errors, 180.0)


class TestComputeGroupedAbsErrors:
    def test_identical_single_cluster(self):
        reconstruction = create_test_reconstruction()

        errors = compute_grouped_abs_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=_single_gt_cluster(reconstruction),
        )

        assert len(errors) == reconstruction.num_images()
        np.testing.assert_allclose(errors, 0.0, atol=1e-10)

    def test_keeps_one_error_per_reconstruction(self):
        # An image registered in n sub-models contributes n errors (not just
        # the smallest). Here every image appears in both sub-models.
        reconstruction = create_test_reconstruction()

        errors = compute_grouped_abs_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction, reconstruction],
            image_name_to_gt_recon_ids=_single_gt_cluster(reconstruction),
        )

        assert len(errors) == 2 * reconstruction.num_images()
        np.testing.assert_allclose(errors, 0.0, atol=1e-10)

    def test_keeps_only_best_cluster(self):
        # Cluster 0 is reconstructed perfectly; cluster 1 is offset. The
        # best-mean cluster (0) is kept intact and cluster 1 is maxed out.
        gt_reconstruction = create_test_reconstruction()
        reconstruction = create_test_reconstruction()

        names = sorted(image.name for image in gt_reconstruction.images.values())
        half = len(names) // 2
        image_name_to_gt_recon_ids = {
            name: (0 if i < half else 1) for i, name in enumerate(names)
        }
        cluster1_names = {name for name in names[half:]}
        for image in reconstruction.images.values():
            if image.name in cluster1_names:
                image.frame.rig_from_world.translation += np.array([5, 5, 5])

        errors = compute_grouped_abs_errors(
            sparse_gt=gt_reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
        )

        assert len(errors) == gt_reconstruction.num_images()
        finite = errors[np.isfinite(errors)]
        # Only the (perfect) best cluster remains finite.
        assert len(finite) == half
        np.testing.assert_allclose(finite, 0.0, atol=1e-10)
        assert int(np.sum(~np.isfinite(errors))) == len(names) - half

    def test_images_without_cluster_id_are_maxed(self):
        # GT images missing from the mapping (cluster id None) are always maxed
        # out, even when perfectly registered.
        reconstruction = create_test_reconstruction()
        names = sorted(image.name for image in reconstruction.images.values())
        half = len(names) // 2
        # Only the second half is mapped (all to cluster 1); the first half is
        # left unmapped (None).
        image_name_to_gt_recon_ids = {name: 1 for name in names[half:]}

        errors = compute_grouped_abs_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
        )

        assert len(errors) == reconstruction.num_images()
        finite = errors[np.isfinite(errors)]
        # The mapped (best) cluster stays finite; unmapped images are maxed.
        assert len(finite) == len(names) - half
        np.testing.assert_allclose(finite, 0.0, atol=1e-10)

    def test_outlier_cluster_is_maxed(self):
        # An outlier is never a selectable cluster, so it is maxed out even
        # though it is perfectly aligned; the real cluster is kept intact.
        reconstruction = create_test_reconstruction()
        names = sorted(image.name for image in reconstruction.images.values())
        outlier = names[0]
        image_name_to_gt_recon_ids = {
            name: (OUTLIER_RECON_ID if name == outlier else 0)
            for name in names
        }

        errors = compute_grouped_abs_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
        )

        assert len(errors) == reconstruction.num_images()
        finite = errors[np.isfinite(errors)]
        assert len(finite) == len(names) - 1
        np.testing.assert_allclose(finite, 0.0, atol=1e-10)

    def test_outlier_present_in_gt_gets_large_error(self):
        # An image that exists in the GT reconstruction and is perfectly
        # registered, but is flagged as an outlier, must still receive a large
        # error: being present and well-aligned does not rescue an outlier.
        reconstruction = create_test_reconstruction()
        gt_names = [image.name for image in reconstruction.images.values()]
        outlier = sorted(gt_names)[0]
        assert outlier in gt_names  # The outlier is present in the GT.
        image_name_to_gt_recon_ids = {
            name: (OUTLIER_RECON_ID if name == outlier else 0)
            for name in gt_names
        }

        errors = compute_grouped_abs_errors(
            sparse_gt=reconstruction,
            sub_models=[reconstruction],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
        )

        # Every GT image is registered exactly once, so errors map 1:1 to
        # gt_names in order.
        assert len(errors) == reconstruction.num_images()
        error_by_name = dict(zip(gt_names, errors))
        assert not np.isfinite(error_by_name[outlier])
        others = [v for n, v in error_by_name.items() if n != outlier]
        np.testing.assert_allclose(others, 0.0, atol=1e-10)

    def test_credits_multiple_clusters_across_sub_models(self):
        # Two sub-models each perfectly reconstruct a different GT cluster and
        # offset the other. Per-sub-model selection credits both clusters,
        # which a single global choice could not do.
        gt_reconstruction = create_test_reconstruction()
        sub_model_0 = create_test_reconstruction()
        sub_model_1 = create_test_reconstruction()

        names = sorted(image.name for image in gt_reconstruction.images.values())
        half = len(names) // 2
        image_name_to_gt_recon_ids = {
            name: (0 if i < half else 1) for i, name in enumerate(names)
        }
        cluster0_names = set(names[:half])
        cluster1_names = set(names[half:])
        # sub_model_0 offsets cluster 1 -> its best cluster is 0.
        for image in sub_model_0.images.values():
            if image.name in cluster1_names:
                image.frame.rig_from_world.translation += np.array([5, 5, 5])
        # sub_model_1 offsets cluster 0 -> its best cluster is 1.
        for image in sub_model_1.images.values():
            if image.name in cluster0_names:
                image.frame.rig_from_world.translation += np.array([5, 5, 5])

        errors = compute_grouped_abs_errors(
            sparse_gt=gt_reconstruction,
            sub_models=[sub_model_0, sub_model_1],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
        )

        # Each GT image is registered in both sub-models -> two errors each.
        assert len(errors) == 2 * gt_reconstruction.num_images()
        finite = errors[np.isfinite(errors)]
        # Exactly one finite error per GT image: cluster 0 via sub_model_0 and
        # cluster 1 via sub_model_1, so both clusters are credited.
        assert len(finite) == gt_reconstruction.num_images()
        np.testing.assert_allclose(finite, 0.0, atol=1e-10)

        # The errors must map back to the right image and sub-model: they are
        # emitted in sparse_gt image order, each image contributing its two
        # errors in sub_model order [sub_model_0, sub_model_1]. A cluster-0
        # image is credited only by sub_model_0 (finite, sub_model_1 maxed) and
        # a cluster-1 image only by sub_model_1 (sub_model_0 maxed, finite).
        gt_names = [image.name for image in gt_reconstruction.images.values()]
        for i, name in enumerate(gt_names):
            err_sub0, err_sub1 = errors[2 * i], errors[2 * i + 1]
            if name in cluster0_names:
                np.testing.assert_allclose(err_sub0, 0.0, atol=1e-10)
                assert not np.isfinite(err_sub1)
            else:
                assert not np.isfinite(err_sub0)
                np.testing.assert_allclose(err_sub1, 0.0, atol=1e-10)

    def test_mismatched_gt_and_estimate_cluster_boundaries(self):
        # GT splits the images 1/3 vs 2/3 while the estimate splits them 2/3 vs
        # 1/3. sub_model_0 spans all of GT cluster 0 plus the cluster-1 images
        # that leaked in; it can credit only one GT cluster, so the leaked
        # cluster-1 images are maxed. sub_model_1 holds the rest of cluster 1
        # and credits them.
        reconstruction = create_test_reconstruction()
        names = sorted(image.name for image in reconstruction.images.values())
        n = len(names)
        third = n // 3
        two_thirds = 2 * n // 3
        image_name_to_gt_recon_ids = {
            name: (0 if i < third else 1) for i, name in enumerate(names)
        }
        sub_model_0 = sub_reconstruction(reconstruction, names[:two_thirds])
        sub_model_1 = sub_reconstruction(reconstruction, names[two_thirds:])
        # Offset the cluster-1 images that leaked into sub_model_0 so its best
        # cluster is unambiguously cluster 0.
        leaked_cluster1 = set(names[third:two_thirds])
        for image in sub_model_0.images.values():
            if image.name in leaked_cluster1:
                image.frame.rig_from_world.translation += np.array([5, 5, 5])

        errors = compute_grouped_abs_errors(
            sparse_gt=reconstruction,
            sub_models=[sub_model_0, sub_model_1],
            image_name_to_gt_recon_ids=image_name_to_gt_recon_ids,
        )

        # Sub-models are disjoint, so each GT image contributes exactly one
        # error, emitted in sparse_gt image order.
        assert len(errors) == reconstruction.num_images()
        gt_names = [image.name for image in reconstruction.images.values()]
        error_by_name = dict(zip(gt_names, errors))
        # Cluster 0 (first third) credited by sub_model_0.
        for name in names[:third]:
            np.testing.assert_allclose(error_by_name[name], 0.0, atol=1e-10)
        # Cluster-1 images that leaked into sub_model_0 are maxed.
        for name in names[third:two_thirds]:
            assert not np.isfinite(error_by_name[name])
        # Remaining cluster 1 credited by sub_model_1.
        for name in names[two_thirds:]:
            np.testing.assert_allclose(error_by_name[name], 0.0, atol=1e-10)
