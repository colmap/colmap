# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path

import pytest


def test_rerun_discards_failed_seed_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import evaluate

    output = tmp_path / "run"
    output.mkdir()
    for name in ("r_s1.pkl", "r_s1.json", "r_s2.pkl", "r_s2.json", "r_s5.pkl"):
        (output / name).write_text("previous")
    args = argparse.Namespace(
        seeds=[1, 2, 3], report_name="r", run_name="run", run_path=tmp_path
    )

    def run(seed_args: argparse.Namespace) -> dict | None:
        if seed_args.random_seed == 1:
            (output / "r_s1.pkl").write_text("partial")
            raise ValueError("failed seed")
        if seed_args.random_seed == 2:
            return None
        (output / "r_s3.pkl").write_text("success")
        return {}

    monkeypatch.setattr(evaluate, "run_once", run)
    with pytest.raises(SystemExit, match="1"):
        evaluate.run_seeds(args)
    assert sorted(path.name for path in output.iterdir()) == [
        "r_s3.pkl",
        "r_s5.pkl",
    ]
