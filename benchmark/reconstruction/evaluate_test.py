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
import json

import evaluate
import pytest


def _make_args(tmp_path, datasets):
    return argparse.Namespace(
        datasets=datasets,
        data_path=tmp_path / "data",
        categories=[],
        scenes=[],
        run_path=tmp_path / "runs",
        run_name="test",
        report_name="report",
        fast=False,
        random_seed=0,
        feature="sift",
        mapper="incremental",
        use_gpu=False,
    )


def test_run_once_rejects_duplicate_bare_dataset_names(tmp_path):
    args = _make_args(tmp_path, ["eth3d", "eth3d:distorted"])

    with pytest.raises(ValueError, match="different --run_name"):
        evaluate.run_once(args)


def test_run_once_preserves_bare_metrics_key_and_records_resolved_variant(
    tmp_path, monkeypatch
):
    calls = []

    class FakeDataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def list_scenes(self):
            return [object()]

    monkeypatch.setattr(evaluate, "DatasetETH3D", FakeDataset)
    monkeypatch.setattr(
        evaluate, "process_scenes", lambda **unused_kwargs: {"dslr": {}}
    )
    monkeypatch.setattr(evaluate, "create_result_table", lambda metrics: "")
    report_dir = tmp_path / "runs" / "test"
    report_dir.mkdir(parents=True)

    metrics = evaluate.run_once(_make_args(tmp_path, ["eth3d"]))

    assert list(metrics) == ["eth3d"]
    assert "variant" not in calls[0]
    metadata = json.loads((report_dir / "report.json").read_text())
    assert metadata["datasets"] == ["eth3d"]
    assert metadata["dataset_variants"] == {"eth3d": "undistorted"}


def test_run_once_passes_explicit_variant(tmp_path, monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def list_scenes(self):
            return [object()]

    monkeypatch.setattr(evaluate, "DatasetETH3D", FakeDataset)
    monkeypatch.setattr(
        evaluate, "process_scenes", lambda **unused_kwargs: {"dslr": {}}
    )
    monkeypatch.setattr(evaluate, "create_result_table", lambda metrics: "")
    report_dir = tmp_path / "runs" / "test"
    report_dir.mkdir(parents=True)

    evaluate.run_once(_make_args(tmp_path, ["eth3d:distorted"]))

    assert calls[0]["variant"] == "distorted"
    metadata = json.loads((report_dir / "report.json").read_text())
    assert metadata["dataset_variants"] == {"eth3d": "distorted"}
