# SPDX-License-Identifier: BSD-3-Clause

import io
import struct
import zipfile
from unittest.mock import Mock

import pytest
import requests

from .package_tartanair_v2 import RemoteZip


def zip_bytes(
    entries: dict[str, bytes], comment: bytes = b"", method: int = 0
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=method) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
        archive.comment = comment
    return buffer.getvalue()


def open_remote(monkeypatch: pytest.MonkeyPatch, data: bytes) -> RemoteZip:
    def get(
        session: requests.Session,
        url: str,
        *,
        headers: dict[str, str],
        timeout: int,
        stream: bool = False,
    ) -> requests.Response:
        first, last = headers["Range"].removeprefix("bytes=").split("-")
        start = int(first) if first else max(0, len(data) - int(last))
        stop = int(last) + 1 if first else len(data)
        assert 0 <= start < stop <= len(data)
        response = requests.Response()
        response.status_code = 206
        response.headers["Content-Range"] = (
            f"bytes {start}-{stop - 1}/{len(data)}"
        )
        response._content = data[start:stop]
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "get", get)
    return RemoteZip("https://example.org/scene.zip")


def test_reject_ignored_range_before_reading_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 200
    raw = Mock()
    response.raw = raw
    get = Mock(return_value=response)
    monkeypatch.setattr(requests.Session, "get", get)
    with pytest.raises(RuntimeError, match="ignored byte range"):
        RemoteZip("https://example.org/scene.zip")
    assert get.call_args.kwargs.get("stream") is True
    raw.close.assert_called_once()
    raw.read.assert_not_called()


def test_empty_archive_and_empty_stored_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert open_remote(monkeypatch, zip_bytes({})).members == {}
    archive = open_remote(monkeypatch, zip_bytes({"empty": b""}))
    assert archive.read("empty")[0] == b""


@pytest.mark.parametrize(
    "comment", [b"note with PK\x05\x06 inside", b"PK\x05\x06" + bytes(18)]
)
def test_directory_signature_in_comment(
    monkeypatch: pytest.MonkeyPatch, comment: bytes
) -> None:
    data = zip_bytes({"pose": b"123"}, comment)
    assert open_remote(monkeypatch, data).read("pose")[0] == b"123"


def test_archive_with_trailing_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    data = zip_bytes({"pose": b"123"}) + bytes(4)
    assert open_remote(monkeypatch, data).read("pose")[0] == b"123"


def test_zip64_end_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1)
    data = zip_bytes({"pose": b"123"})
    assert open_remote(monkeypatch, data).read("pose")[0] == b"123"


def test_legacy_member_name(monkeypatch: pytest.MonkeyPatch) -> None:
    data = bytearray(zip_bytes({"x": b"frame"}))
    directory = data.index(b"PK\x01\x02")
    data[30] = data[directory + 46] = 0x82  # "é" in CP437.
    assert open_remote(monkeypatch, bytes(data)).read("é")[0] == b"frame"


def test_truncated_directory_record(monkeypatch: pytest.MonkeyPatch) -> None:
    data = bytearray(zip_bytes({"frame": b"123"}))
    directory = data.index(b"PK\x01\x02")
    struct.pack_into("<H", data, directory + 32, 13)  # Missing comment.
    with pytest.raises(RuntimeError, match="Truncated ZIP directory entry"):
        open_remote(monkeypatch, bytes(data))


def test_utf8_deflated_member(monkeypatch: pytest.MonkeyPatch) -> None:
    data = zip_bytes({"轨迹.txt": b"1 2 3"}, method=zipfile.ZIP_DEFLATED)
    assert open_remote(monkeypatch, data).read("轨迹.txt")[0] == b"1 2 3"
