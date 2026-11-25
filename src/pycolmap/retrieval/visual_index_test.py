# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any

import pytest

import pycolmap


def test_image_score() -> None:
    score = pycolmap.ImageScore()
    assert score.image_id == -1
    assert score.score == 0.0
    with pytest.raises(AttributeError):
        score.image_id = 1


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
    assert [o.max_num_images for o in index.query_options] == [5, 5]

    path = tmp_path / "index.bin"
    index.write(path)
    assert index.path == path
