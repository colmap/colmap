# SPDX-License-Identifier: BSD-3-Clause

import zipfile
from pathlib import Path

import pytest


def test_imc2024_relative_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("py7zr")
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import download

    monkeypatch.chdir(tmp_path)
    data_path = Path("data/imc2024")
    data_path.mkdir(parents=True)
    with zipfile.ZipFile(
        data_path / "image-matching-challenge-2024.zip", "w"
    ) as archive:
        archive.writestr("train/scene/images/one.jpg", b"image")
    monkeypatch.setattr(download.subprocess, "check_call", lambda cmd: 0)
    download.download_imc2024(data_path)
    image = data_path / "train/all/scene/images/one.jpg"
    assert image.read_bytes() == b"image"
