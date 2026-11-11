# SPDX-License-Identifier: BSD-3-Clause

"""Round trips of the cusift option dataclasses and of the parent options
that hold them.

The cusift options have no C++ operator==. So each test compares todict(),
and also reads the non-default cusift fields directly: todict() and pickle
share one attribute list, so todict() alone misses a field left out of it.
"""

import copy
import pickle
from collections.abc import Callable
from typing import Any

import pytest

import pycolmap


def make_cusift_extraction() -> pycolmap.CusiftExtractionOptions:
    options = pycolmap.CusiftExtractionOptions()
    options.num_octaves = 5
    return options


def make_cusift_matching() -> pycolmap.CusiftMatchingOptions:
    options = pycolmap.CusiftMatchingOptions()
    options.cross_check_reverse_ratio = False
    options.distance_type = pycolmap.CusiftDistanceType.ANGULAR
    return options


def make_extraction() -> pycolmap.FeatureExtractionOptions:
    options = pycolmap.FeatureExtractionOptions(
        type=pycolmap.FeatureExtractorType.CUSIFT
    )
    options.cusift.num_octaves = 5
    return options


def make_matching() -> pycolmap.FeatureMatchingOptions:
    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.CUSIFT_GEMM
    )
    options.cusift.cross_check_reverse_ratio = False
    options.cusift.distance_type = pycolmap.CusiftDistanceType.ANGULAR
    return options


def extraction_fields(options: pycolmap.CusiftExtractionOptions) -> tuple:
    return (options.num_octaves,)


def matching_fields(options: pycolmap.CusiftMatchingOptions) -> tuple:
    return (options.cross_check_reverse_ratio, options.distance_type)


def pickle_round_trip(options: Any) -> Any:
    return pickle.loads(pickle.dumps(options))


def dict_round_trip(options: Any) -> Any:
    return type(options)(options.todict())


# Each case: how to make the options, and how to read their cusift fields.
CASES: dict[str, tuple[Callable[[], Any], Callable[[Any], tuple]]] = {
    "cusift_extraction": (make_cusift_extraction, extraction_fields),
    "cusift_matching": (make_cusift_matching, matching_fields),
    "extraction": (make_extraction, lambda o: extraction_fields(o.cusift)),
    "matching": (make_matching, lambda o: matching_fields(o.cusift)),
}
ROUND_TRIPS: dict[str, Callable[[Any], Any]] = {
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
    "pickle": pickle_round_trip,
    "todict": dict_round_trip,
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize(
    "round_trip", ROUND_TRIPS.values(), ids=ROUND_TRIPS.keys()
)
def test_round_trip_keeps_cusift_fields(
    case: tuple[Callable[[], Any], Callable[[Any], tuple]],
    round_trip: Callable[[Any], Any],
) -> None:
    make_options, read_fields = case
    options = make_options()

    restored = round_trip(options)

    assert restored is not options
    assert read_fields(restored) == read_fields(make_options())
    assert restored.todict() == options.todict()


@pytest.mark.parametrize("copy_options", [copy.copy, copy.deepcopy])
def test_copied_extraction_options_own_their_cusift_options(
    copy_options: Callable[[Any], Any],
) -> None:
    options = make_extraction()

    copied = copy_options(options)
    copied.cusift.num_octaves = 7

    assert copied.cusift.num_octaves == 7
    assert options.cusift.num_octaves == 5


@pytest.mark.parametrize("copy_options", [copy.copy, copy.deepcopy])
def test_copied_matching_options_own_their_cusift_options(
    copy_options: Callable[[Any], Any],
) -> None:
    options = make_matching()

    copied = copy_options(options)
    copied.cusift.distance_type = pycolmap.CusiftDistanceType.L2

    assert copied.cusift.distance_type == pycolmap.CusiftDistanceType.L2
    assert options.cusift.distance_type == pycolmap.CusiftDistanceType.ANGULAR


def test_extraction_mergedict_sets_nested_cusift_fields() -> None:
    options = pycolmap.FeatureExtractionOptions()

    options.mergedict({"cusift": {"num_octaves": 6}})

    assert options.cusift.num_octaves == 6


def test_matching_mergedict_sets_nested_cusift_fields() -> None:
    options = pycolmap.FeatureMatchingOptions()

    options.mergedict(
        {
            "cusift": {
                "cross_check_reverse_ratio": False,
                "distance_type": pycolmap.CusiftDistanceType.ANGULAR,
            }
        }
    )

    assert options.cusift.cross_check_reverse_ratio is False
    assert options.cusift.distance_type == pycolmap.CusiftDistanceType.ANGULAR
