# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pycolmap


def test_image_score() -> None:
    score = pycolmap.ImageScore()
    assert score.image_id == -1
    assert score.score == 0.0
    with pytest.raises(AttributeError):
        score.image_id = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("options_cls", "values"),
    [
        (
            pycolmap.VisualIndex.IndexOptions,
            {"num_neighbors": 5, "num_checks": 7, "num_threads": 3},
        ),
        (
            pycolmap.VisualIndex.QueryOptions,
            {
                "max_num_images": 5,
                "num_neighbors": 6,
                "num_checks": 7,
                "num_threads": 3,
                "num_images_after_verification": 9,
            },
        ),
        (
            pycolmap.VisualIndex.BuildOptions,
            {
                "num_visual_words": 5,
                "num_iterations": 6,
                "num_rounds": 2,
                "num_checks": 7,
                "num_threads": 3,
            },
        ),
    ],
)
def test_visual_index_options(
    options_cls: type, values: dict[str, int]
) -> None:
    options = options_cls(**values)
    for name, value in values.items():
        assert getattr(options, name) == value


def test_visual_index_create() -> None:
    assert pycolmap.VisualIndex.create().desc_dim() == 128
    assert pycolmap.VisualIndex.create(32).desc_dim() == 32
    index = pycolmap.VisualIndex.create(128, 64)
    assert index.desc_dim() == 128
    assert index.embedding_dim() == 64
    assert index.num_visual_words() == 0
    assert index.num_images() == 0


def test_visual_index_python_override(tmp_path: Path) -> None:
    class CustomIndex(pycolmap.VisualIndex):
        def Query(self, options: Any, *args: Any) -> None:
            self.query_options.append(options)

        def Write(self, path: Path) -> None:
            self.path = path

    index = CustomIndex()
    index.query_options = []
    options = pycolmap.VisualIndex.QueryOptions(max_num_images=5)
    keypoints = pycolmap.FeatureKeypoints()
    descriptors = pycolmap.FeatureDescriptorsFloat()
    index.query(options, descriptors, [])
    index.query(options, keypoints, descriptors, [])
    assert index.query(options, descriptors) == []
    assert index.query(options, keypoints, descriptors) == []
    assert [o.max_num_images for o in index.query_options] == [5, 5, 5, 5]

    path = tmp_path / "index.bin"
    index.write(path)
    assert index.path == path


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
