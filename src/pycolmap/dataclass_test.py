import copy
import pickle

import pycolmap

# --- Tests on RANSACOptions (flat dataclass) ---


def test_ransac_options_summary():
    options = pycolmap.RANSACOptions()
    summary = options.summary()
    assert isinstance(summary, str)
    assert "RANSACOptions" in summary


def test_ransac_options_summary_write_type():
    options = pycolmap.RANSACOptions()
    summary = options.summary(write_type=True)
    assert isinstance(summary, str)
    assert "RANSACOptions" in summary


def test_ransac_options_repr():
    options = pycolmap.RANSACOptions()
    representation = repr(options)
    assert isinstance(representation, str)
    assert len(representation) > 0


def test_ransac_options_todict():
    options = pycolmap.RANSACOptions()
    dictionary = options.todict()
    assert isinstance(dictionary, dict)
    assert "max_error" in dictionary
    assert "confidence" in dictionary


def test_ransac_options_todict_recursive():
    options = pycolmap.RANSACOptions()
    dictionary = options.todict(recursive=True)
    assert isinstance(dictionary, dict)
    assert "max_error" in dictionary


def test_ransac_options_mergedict():
    options = pycolmap.RANSACOptions()
    options.mergedict({"max_error": 12.0, "confidence": 0.99})
    assert options.max_error == 12.0
    assert options.confidence == 0.99


def test_ransac_options_copy():
    options = pycolmap.RANSACOptions()
    options.max_error = 7.5
    copied = copy.copy(options)
    assert copied.max_error == 7.5
    assert copied is not options


def test_ransac_options_deepcopy():
    options = pycolmap.RANSACOptions()
    options.max_error = 3.3
    deep = copy.deepcopy(options)
    assert deep.max_error == 3.3
    assert deep is not options


def test_ransac_options_eq():
    options_a = pycolmap.RANSACOptions()
    options_b = pycolmap.RANSACOptions()
    assert options_a == options_b
    options_b.max_error = 999.0
    assert options_a != options_b


def test_ransac_options_pickle_roundtrip():
    options = pycolmap.RANSACOptions()
    options.max_error = 5.5
    options.confidence = 0.95
    data = pickle.dumps(options)
    restored = pickle.loads(data)
    assert restored.max_error == 5.5
    assert restored.confidence == 0.95


def test_ransac_options_dict_constructor():
    options = pycolmap.RANSACOptions({"max_error": 11.0, "confidence": 0.98})
    assert options.max_error == 11.0
    assert options.confidence == 0.98


def test_ransac_options_kwargs_constructor():
    options = pycolmap.RANSACOptions(max_error=22.0, min_num_trials=200)
    assert options.max_error == 22.0
    assert options.min_num_trials == 200


# --- Tests on IncrementalPipelineOptions (nested dataclass) ---


def test_incremental_pipeline_options_todict_recursive():
    options = pycolmap.IncrementalPipelineOptions()
    dictionary = options.todict(recursive=True)
    assert isinstance(dictionary, dict)
    assert "min_num_matches" in dictionary


def test_incremental_pipeline_options_mergedict():
    options = pycolmap.IncrementalPipelineOptions()
    options.mergedict({"min_num_matches": 30})
    assert options.min_num_matches == 30


def test_incremental_pipeline_options_summary():
    options = pycolmap.IncrementalPipelineOptions()
    summary = options.summary()
    assert isinstance(summary, str)
    assert "IncrementalPipelineOptions" in summary


def test_incremental_pipeline_options_pickle():
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = 42
    data = pickle.dumps(options)
    restored = pickle.loads(data)
    assert restored.min_num_matches == 42


# --- Tests on Rigid3d ---


def test_rigid3d_summary():
    rigid = pycolmap.Rigid3d()
    summary = rigid.summary()
    assert isinstance(summary, str)
    assert "Rigid3d" in summary


def test_rigid3d_todict():
    rigid = pycolmap.Rigid3d()
    dictionary = rigid.todict()
    assert isinstance(dictionary, dict)


def test_rigid3d_pickle_roundtrip():
    rigid = pycolmap.Rigid3d()
    data = pickle.dumps(rigid)
    restored = pickle.loads(data)
    assert isinstance(restored, pycolmap.Rigid3d)
    assert restored == rigid
