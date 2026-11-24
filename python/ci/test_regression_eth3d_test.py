# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from types import ModuleType
from urllib.error import ContentTooShortError, HTTPError

import pytest


@dataclass
class DownloadServer:
    file_path: Path
    url: str = ""
    responses: list[tuple[int, bytes, int]] = field(default_factory=list)
    destination_exists: list[bool] = field(default_factory=list)


@pytest.fixture
def download_server(tmp_path: Path) -> Iterator[DownloadServer]:
    download = DownloadServer(tmp_path / "dataset.7z")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            download.destination_exists.append(download.file_path.exists())
            status, body, length = (
                download.responses.pop(0)
                if download.responses
                else (500, b"unexpected request", 18)
            )
            self.send_response(status)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    download.url = f"http://127.0.0.1:{server.server_port}/dataset.7z"
    thread = Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    try:
        yield download
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def regression(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import test_regression_eth3d

    return test_regression_eth3d


def test_download_retries_without_publishing_partial_file(
    regression: ModuleType, download_server: DownloadServer
) -> None:
    download = download_server
    complete = b"complete archive contents"
    download.responses.extend(
        [(200, b"partial", 100), (200, complete, len(complete))]
    )
    regression.download_file(
        download.url, str(download.file_path), max_retries=2
    )
    assert download.file_path.read_bytes() == complete
    assert download.destination_exists == [False, False]
    assert list(download.file_path.parent.iterdir()) == [download.file_path]


@pytest.mark.parametrize(
    ("status", "error"), [(200, ContentTooShortError), (503, HTTPError)]
)
def test_failed_download_raises_and_can_be_retried(
    regression: ModuleType,
    download_server: DownloadServer,
    status: int,
    error: type[Exception],
) -> None:
    download = download_server
    download.responses.extend([(status, b"partial", 100)] * 2)
    with pytest.raises(error):
        regression.download_file(
            download.url, str(download.file_path), max_retries=2
        )
    assert not download.file_path.exists()
    assert list(download.file_path.parent.iterdir()) == []
    assert download.destination_exists == [False, False]

    complete = b"complete archive contents"
    download.responses.append((200, complete, len(complete)))
    regression.download_file(download.url, str(download.file_path))
    assert download.file_path.read_bytes() == complete
    assert download.destination_exists == [False, False, False]
    assert list(download.file_path.parent.iterdir()) == [download.file_path]


def test_download_reuses_existing_file(
    regression: ModuleType, download_server: DownloadServer
) -> None:
    download = download_server
    download.file_path.write_bytes(b"cached archive")
    regression.download_file(download.url, str(download.file_path))
    assert download.file_path.read_bytes() == b"cached archive"
    assert download.destination_exists == []
