# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap


def test_image_score_class_exists() -> None:
    score = pycolmap.ImageScore()
    assert score is not None


def test_image_score_readonly_image_id() -> None:
    score = pycolmap.ImageScore()
    assert isinstance(score.image_id, int)


def test_image_score_readonly_score() -> None:
    score = pycolmap.ImageScore()
    assert isinstance(score.score, (int, float))


def test_visual_index_index_options_init() -> None:
    options = pycolmap.VisualIndex.IndexOptions()
    assert options is not None


def test_visual_index_index_options_num_neighbors() -> None:
    options = pycolmap.VisualIndex.IndexOptions()
    options.num_neighbors = 5
    assert options.num_neighbors == 5


def test_visual_index_index_options_num_checks() -> None:
    options = pycolmap.VisualIndex.IndexOptions()
    options.num_checks = 128
    assert options.num_checks == 128


def test_visual_index_index_options_num_threads() -> None:
    options = pycolmap.VisualIndex.IndexOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_visual_index_query_options_init() -> None:
    options = pycolmap.VisualIndex.QueryOptions()
    assert options is not None


def test_visual_index_query_options_max_num_images() -> None:
    options = pycolmap.VisualIndex.QueryOptions()
    options.max_num_images = 50
    assert options.max_num_images == 50


def test_visual_index_query_options_num_neighbors() -> None:
    options = pycolmap.VisualIndex.QueryOptions()
    options.num_neighbors = 3
    assert options.num_neighbors == 3


def test_visual_index_query_options_num_checks() -> None:
    options = pycolmap.VisualIndex.QueryOptions()
    options.num_checks = 64
    assert options.num_checks == 64


def test_visual_index_query_options_num_threads() -> None:
    options = pycolmap.VisualIndex.QueryOptions()
    options.num_threads = 2
    assert options.num_threads == 2


def test_visual_index_query_options_num_images_after_verification() -> None:
    options = pycolmap.VisualIndex.QueryOptions()
    options.num_images_after_verification = 10
    assert options.num_images_after_verification == 10


def test_visual_index_build_options_init() -> None:
    options = pycolmap.VisualIndex.BuildOptions()
    assert options is not None


def test_visual_index_build_options_num_visual_words() -> None:
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_visual_words = 1024
    assert options.num_visual_words == 1024


def test_visual_index_build_options_num_iterations() -> None:
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_iterations = 5
    assert options.num_iterations == 5


def test_visual_index_build_options_num_rounds() -> None:
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_rounds = 3
    assert options.num_rounds == 3


def test_visual_index_build_options_num_checks() -> None:
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_checks = 256
    assert options.num_checks == 256


def test_visual_index_build_options_num_threads() -> None:
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_threads = 8
    assert options.num_threads == 8


def test_visual_index_create() -> None:
    index = pycolmap.VisualIndex.create(128, 64)
    assert index is not None
    assert index.num_visual_words() == 0
    assert index.num_images() == 0


def test_visual_index_create_with_descriptor_dimension() -> None:
    for desc_dim in (32, 128, 256):
        index = pycolmap.VisualIndex.create(desc_dim)
        assert index.desc_dim() == desc_dim


@pytest.mark.parametrize("spatial_verification", [False, True])
def test_visual_index_query_returns_scores(spatial_verification: bool) -> None:
    rng = np.random.default_rng(0)
    training_data = rng.standard_normal((1000, 128)).astype(np.float32)
    training_descriptors = pycolmap.FeatureDescriptorsFloat(
        pycolmap.FeatureExtractorType.SIFT, training_data
    )
    index = pycolmap.VisualIndex.create(128, 64)
    index.build(
        pycolmap.VisualIndex.BuildOptions(
            num_visual_words=100,
            num_iterations=5,
            num_rounds=1,
            num_threads=1,
        ),
        training_descriptors,
    )
    keypoints = pycolmap.FeatureKeypoints(
        [
            pycolmap.FeatureKeypoint(x=10.0 * (i % 10), y=10.0 * (i // 10))
            for i in range(50)
        ]
    )
    descriptors = pycolmap.FeatureDescriptorsFloat(
        pycolmap.FeatureExtractorType.SIFT, training_data[:50]
    )
    other_descriptors = pycolmap.FeatureDescriptorsFloat(
        pycolmap.FeatureExtractorType.SIFT,
        np.concatenate((training_data[:10], training_data[50:90])),
    )
    index_options = pycolmap.VisualIndex.IndexOptions(num_threads=1)
    index.add(index_options, 7, keypoints, descriptors)
    index.add(index_options, 9, keypoints, other_descriptors)
    index.prepare()
    query_options = pycolmap.VisualIndex.QueryOptions(
        num_neighbors=1,
        num_threads=1,
        num_images_after_verification=2 if spatial_verification else 0,
    )
    output = [pycolmap.ImageScore()]
    if spatial_verification:
        result = index.query(query_options, keypoints, descriptors, output)
    else:
        result = index.query(query_options, descriptors, output)
    assert result is None
    assert [score.image_id for score in output] == [7, 9]

    if spatial_verification:
        scores = index.query(query_options, keypoints, descriptors)
    else:
        scores = index.query(query_options, descriptors)
    assert [score.image_id for score in scores] == [7, 9]
    assert scores[0].score > scores[1].score
    np.testing.assert_allclose(
        [score.score for score in output], [score.score for score in scores]
    )

    query_options.max_num_images = 1
    if spatial_verification:
        limited_scores = index.query(query_options, keypoints, descriptors)
    else:
        limited_scores = index.query(query_options, descriptors)
    assert [score.image_id for score in limited_scores] == [7]

    if spatial_verification:
        index.query(query_options, keypoints, descriptors, output)
    else:
        index.query(query_options, descriptors, output)
    assert [score.image_id for score in output] == [7]
