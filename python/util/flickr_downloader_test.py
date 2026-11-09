# SPDX-License-Identifier: BSD-3-Clause

import io
import runpy
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import pytest


def _photo_page_response(url: str) -> io.BytesIO:
    return io.BytesIO(
        f'<rsp stat="ok"><photos pages="1"><photo id="12" secret="abc" '
        f'url_l="{url}"/></photos></rsp>'.encode()
    )


def test_download_photo_from_page(tmp_path: Path) -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    url = "https://example.org/image.jpg?size=large"
    response = _photo_page_response(url)
    with patch.object(urllib.request, "urlopen", return_value=response):
        metadata, photos = downloader["parse_page"](1, "key", "scene", 0, 1)
    assert metadata["pages"] == "1"
    assert len(photos) == 1
    assert response.closed

    def retrieve_photo(url: str, filename: str) -> None:
        Path(filename).write_bytes(b"complete photo")

    with patch.object(
        urllib.request, "urlretrieve", side_effect=retrieve_photo
    ) as retrieve:
        downloader["PhotoDownloader"](str(tmp_path))(photos[0])
    assert retrieve.call_count == 1
    assert retrieve.call_args.args[0] == url
    assert (tmp_path / "12_abc.jpg").read_bytes() == b"complete photo"


def test_parse_page_retries_after_timeout() -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    url = "https://example.org/image.jpg?size=large"
    response = _photo_page_response(url)
    with patch.object(
        urllib.request,
        "urlopen",
        side_effect=[TimeoutError, response],
    ) as urlopen_mock:
        metadata, photos = downloader["parse_page"](1, "key", "scene", 0, 1)
    assert metadata["pages"] == "1"
    assert len(photos) == 1
    assert response.closed
    assert urlopen_mock.call_count == 2


def test_parse_page_returns_empty_after_repeated_timeouts() -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    with patch.object(
        urllib.request, "urlopen", side_effect=TimeoutError
    ) as urlopen_mock:
        metadata, photos = downloader["parse_page"](1, "key", "scene", 0, 1)
    assert metadata == {
        "pages": "0",
        "total": "0",
        "page": "0",
        "perpage": "0",
    }
    assert photos == ()
    assert urlopen_mock.call_count == downloader["MAX_PAGE_REQUESTS"]


def test_search_query_round_trip() -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    text = "café & bridge #2 + hill"
    url = downloader["compose_url"](2, "key", text, 10, 20)
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert query["text"] == [text]
    assert query["page"] == ["2"]
    assert query["max_upload_date"] == ["20"]


def test_only_small_photo(tmp_path: Path) -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    download = downloader["PhotoDownloader"](str(tmp_path))
    photo = {"id": "12", "secret": "s", "url_n": "https://e.org/n.jpg"}
    with patch.object(
        urllib.request,
        "urlretrieve",
        side_effect=lambda url, path: Path(path).write_bytes(b"small"),
    ):
        download(photo)
    assert (tmp_path / "12_s.jpg").read_bytes() == b"small"


def test_failed_download_can_be_retried(tmp_path: Path) -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    download = downloader["PhotoDownloader"](str(tmp_path))
    photo = {"id": "12", "secret": "s", "url_o": "https://e.org/o.png"}

    def truncated(url: str, path: str) -> None:
        Path(path).write_bytes(b"short")
        raise urllib.error.ContentTooShortError("truncated", (path, Message()))

    with patch.object(urllib.request, "urlretrieve", side_effect=truncated):
        download(photo)
    assert not list(tmp_path.iterdir())
    with patch.object(
        urllib.request,
        "urlretrieve",
        side_effect=lambda url, path: Path(path).write_bytes(b"complete"),
    ):
        download(photo)
    assert (tmp_path / "12_s.png").read_bytes() == b"complete"


def test_service_error_includes_reason() -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    error = b'<rsp stat="fail"><err code="100" msg="Invalid API Key"/></rsp>'
    with patch.object(
        urllib.request, "urlopen", return_value=io.BytesIO(error)
    ):
        with pytest.raises(OSError, match="100.*Invalid API Key"):
            downloader["parse_page"](1, "key", "scene", 0, 1)


def test_create_photo_directory(tmp_path: Path) -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    image_path = tmp_path / "new" / "photos"
    downloader["PhotoDownloader"](str(image_path))
    assert image_path.is_dir()
