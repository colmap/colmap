# SPDX-License-Identifier: BSD-3-Clause

import io
import runpy
import urllib.request
from pathlib import Path
from unittest.mock import patch


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
    with patch.object(urllib.request, "urlretrieve") as retrieve:
        downloader["PhotoDownloader"](str(tmp_path))(photos[0])
    retrieve.assert_called_once_with(url, str(tmp_path / "12_abc.jpg"))


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
