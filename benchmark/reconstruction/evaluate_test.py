# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import pickle
from pathlib import Path
from unittest.mock import Mock

import pytest


def test_skip_datasets_without_matching_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import evaluate

    empty = Mock()
    empty.return_value.list_scenes.return_value = []
    match = Mock()
    match.return_value.list_scenes.return_value = [Mock()]
    monkeypatch.setattr(
        evaluate, "_dataset_classes", lambda: {"empty": empty, "match": match}
    )
    monkeypatch.setattr(evaluate, "process_scenes", lambda **kwargs: {"c": {}})
    monkeypatch.setattr(evaluate, "create_result_table", lambda _: "")
    args = argparse.Namespace(
        datasets=["empty", "match"],
        categories=[],
        scenes=[],
        data_path=tmp_path,
        run_path=tmp_path,
        run_name="run",
        report_name="report",
        fast=False,
        random_seed=0,
        feature="sift",
        mapper="global",
        use_gpu=False,
        uncalibrated=False,
        single_view_calibration=False,
    )
    output = tmp_path / "run"
    output.mkdir()
    assert evaluate.run_once(args) == {"match": {"c": {}}}
    assert list(pickle.loads((output / "report.pkl").read_bytes())) == ["match"]
    metadata = json.loads((output / "report.json").read_text())
    assert metadata["datasets"] == ["match"]

    args.datasets = ["empty"]
    args.report_name = "empty"
    assert evaluate.run_once(args) is None
    assert not (output / "empty.pkl").exists()
