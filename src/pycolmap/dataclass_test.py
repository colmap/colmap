# SPDX-License-Identifier: BSD-3-Clause

import copy
import pickle

import numpy as np
import pytest

import pycolmap

# --- Tests on RANSACOptions (flat dataclass) ---


def test_ransac_options_summary() -> None:
    options = pycolmap.RANSACOptions()
    summary = options.summary()
    assert isinstance(summary, str)
    assert "RANSACOptions" in summary


def test_ransac_options_summary_write_type() -> None:
    options = pycolmap.RANSACOptions()
    summary = options.summary(write_type=True)
    assert isinstance(summary, str)
    assert "RANSACOptions" in summary


def test_ransac_options_repr() -> None:
    options = pycolmap.RANSACOptions()
    representation = repr(options)
    assert isinstance(representation, str)
    assert len(representation) > 0


def test_ransac_options_todict() -> None:
    options = pycolmap.RANSACOptions()
    dictionary = options.todict()
    assert isinstance(dictionary, dict)
    assert "max_error" in dictionary
    assert "confidence" in dictionary


def test_ransac_options_todict_recursive() -> None:
    options = pycolmap.RANSACOptions()
    dictionary = options.todict(recursive=True)
    assert isinstance(dictionary, dict)
    assert "max_error" in dictionary


def test_ransac_options_mergedict() -> None:
    options = pycolmap.RANSACOptions()
    options.mergedict({"max_error": 12.0, "confidence": 0.99})
    assert options.max_error == 12.0
    assert options.confidence == 0.99


def test_ransac_options_copy() -> None:
    options = pycolmap.RANSACOptions()
    options.max_error = 7.5
    copied = copy.copy(options)
    assert copied.max_error == 7.5
    assert copied is not options


def test_ransac_options_deepcopy() -> None:
    options = pycolmap.RANSACOptions()
    options.max_error = 3.3
    deep = copy.deepcopy(options)
    assert deep.max_error == 3.3
    assert deep is not options


def test_ransac_options_eq() -> None:
    options_a = pycolmap.RANSACOptions()
    options_b = pycolmap.RANSACOptions()
    assert options_a == options_b
    options_b.max_error = 999.0
    assert options_a != options_b


def test_ransac_options_pickle_roundtrip() -> None:
    options = pycolmap.RANSACOptions()
    options.max_error = 5.5
    options.confidence = 0.95
    data = pickle.dumps(options)
    restored = pickle.loads(data)
    assert restored.max_error == 5.5
    assert restored.confidence == 0.95


def test_ransac_options_dict_constructor() -> None:
    options = pycolmap.RANSACOptions({"max_error": 11.0, "confidence": 0.98})
    assert options.max_error == 11.0
    assert options.confidence == 0.98


def test_ransac_options_kwargs_constructor() -> None:
    options = pycolmap.RANSACOptions(max_error=22.0, min_num_trials=200)
    assert options.max_error == 22.0
    assert options.min_num_trials == 200


# --- Tests on IncrementalPipelineOptions (nested dataclass) ---


def test_incremental_pipeline_options_todict_recursive() -> None:
    options = pycolmap.IncrementalPipelineOptions()
    dictionary = options.todict(recursive=True)
    assert isinstance(dictionary, dict)
    assert "min_num_matches" in dictionary


def test_incremental_pipeline_options_mergedict() -> None:
    options = pycolmap.IncrementalPipelineOptions()
    options.mergedict({"min_num_matches": 30})
    assert options.min_num_matches == 30


def test_incremental_pipeline_options_summary() -> None:
    options = pycolmap.IncrementalPipelineOptions()
    summary = options.summary()
    assert isinstance(summary, str)
    assert "IncrementalPipelineOptions" in summary


def test_incremental_pipeline_options_pickle() -> None:
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = 42
    data = pickle.dumps(options)
    restored = pickle.loads(data)
    assert restored.min_num_matches == 42


# --- Tests on Rigid3d ---


def test_rigid3d_summary() -> None:
    rigid = pycolmap.Rigid3d()
    summary = rigid.summary()
    assert isinstance(summary, str)
    assert "Rigid3d" in summary


def test_rigid3d_todict() -> None:
    rigid = pycolmap.Rigid3d()
    dictionary = rigid.todict()
    assert isinstance(dictionary, dict)


def test_rigid3d_pickle_roundtrip() -> None:
    rigid = pycolmap.Rigid3d()
    data = pickle.dumps(rigid)
    restored = pickle.loads(data)
    assert isinstance(restored, pycolmap.Rigid3d)
    assert restored == rigid


def test_dataclass_equality_numpy_rotation() -> None:
    left = pycolmap.Rotation3d()
    right = pycolmap.Rotation3d()
    assert (left == right) is True
    assert (left != right) is False
    right.quat = np.array([0.0, 0.0, 1.0, 0.0])
    assert (left == right) is False
    assert (left != right) is True
    assert (left == object()) is False


def test_dataclass_equality_numpy_box() -> None:
    left = pycolmap.AlignedBox3d(np.zeros(3), np.ones(3))
    right = pycolmap.AlignedBox3d(np.zeros(3), np.ones(3))
    assert left == right
    right.max = np.array([1.0, 2.0, 1.0])
    assert left != right


@pytest.mark.parametrize("num_matches", [0, 1, 3])
def test_dataclass_equality_numpy_matches(num_matches: int) -> None:
    matches = np.arange(2 * num_matches, dtype=np.uint32).reshape(-1, 2)
    left = pycolmap.TwoViewGeometry(inlier_matches=matches)
    right = pycolmap.TwoViewGeometry(inlier_matches=matches.copy())
    assert left == right
    right.inlier_matches = np.zeros((num_matches + 1, 2), dtype=np.uint32)
    assert left != right
    if num_matches:
        right.inlier_matches = matches + 1
        assert left != right


def test_dataclass_equality_numpy_matrix() -> None:
    left = pycolmap.TwoViewGeometry(E=np.eye(3))
    right = pycolmap.TwoViewGeometry(E=np.eye(3))
    assert left == right
    right.E = np.diag([1.0, 2.0, 1.0])
    assert left != right
    right.E = None
    assert left != right
    assert right != left


def test_dataclass_equality_nested_numpy_attributes() -> None:
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    left = pycolmap.TwoViewGeometry(
        cam2_from_cam1=pycolmap.Rigid3d(), camera1=camera
    )
    right = copy.deepcopy(left)
    assert left == right
    assert right.cam2_from_cam1 is not None
    right.cam2_from_cam1.translation = np.array([1.0, 0.0, 0.0])
    assert left != right
    right.cam2_from_cam1 = pycolmap.Rigid3d()
    assert left == right
    assert right.camera1 is not None
    right.camera1.params = [600.0, 500.0, 512.0, 384.0]
    assert left != right


# --- Tests on dataclasses with enum and string attributes ---


def test_incremental_mapper_options_repr_uses_qualified_enums() -> None:
    options = pycolmap.IncrementalMapperOptions()
    representation = repr(options)
    assert (
        "image_selection_method=ImageSelectionMethod.MIN_UNCERTAINTY"
        in representation
    )


def test_incremental_mapper_options_summary_uses_qualified_enums() -> None:
    options = pycolmap.IncrementalMapperOptions()
    summary = options.summary()
    assert "ImageSelectionMethod.MIN_UNCERTAINTY" in summary


def test_ceres_ba_options_repr_quotes_strings() -> None:
    options = pycolmap.CeresBundleAdjustmentOptions()
    assert "gpu_index='-1'" in repr(options)
    # repr() escapes strings that manual single-quote wrapping would corrupt.
    options.gpu_index = "a'b"
    assert 'gpu_index="a\'b"' in repr(options)


def test_dataclass_docstrings_show_qualified_enum_defaults() -> None:
    doc = pycolmap.IncrementalMapperOptions.image_selection_method.__doc__
    assert doc is not None and "ImageSelectionMethod.MIN_UNCERTAINTY" in doc


def test_dataclass_docstrings_quote_string_defaults() -> None:
    doc = pycolmap.CeresBundleAdjustmentOptions.gpu_index.__doc__
    assert doc is not None and "default: '-1'" in doc
