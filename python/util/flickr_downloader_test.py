# SPDX-License-Identifier: BSD-3-Clause

import io
import runpy
import urllib.request
from pathlib import Path
from unittest.mock import patch


def test_download_photo_from_page(tmp_path: Path) -> None:
    downloader = runpy.run_path(
        str(Path(__file__).with_name("flickr_downloader.py"))
    )
    url = "https://example.org/image.jpg?size=large"
    response = io.BytesIO(
        f'<rsp stat="ok"><photos pages="1"><photo id="12" secret="abc" '
        f'url_l="{url}"/></photos></rsp>'.encode()
    )
    with patch.object(urllib.request, "urlopen", return_value=response):
        metadata, photos = downloader["parse_page"](1, "key", "scene", 0, 1)
    assert metadata["pages"] == "1"
    with patch.object(urllib.request, "urlretrieve") as retrieve:
        downloader["PhotoDownloader"](str(tmp_path))(photos[0])
    retrieve.assert_called_once_with(url, str(tmp_path / "12_abc.jpg"))
