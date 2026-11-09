# SPDX-License-Identifier: BSD-3-Clause

import argparse
import datetime
import multiprocessing
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Final

PER_PAGE: Final[int] = 500
SORT: Final[str] = "date-posted-desc"
SEARCH_URL: Final[str] = "https://api.flickr.com/services/rest/"
MAX_PAGE_REQUESTS: Final[int] = 5
MAX_PAGE_TIMEOUT: Final[int] = 20
MAX_IMAGE_REQUESTS: Final[int] = 3
TIME_SKIP: Final[int] = 24 * 60 * 60
MAX_DATE: Final[float] = time.time()
MIN_DATE: Final[float] = MAX_DATE - TIME_SKIP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_text", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--num_procs", type=int, default=10)
    parser.add_argument("--max_days_without_image", type=int, default=365)
    args = parser.parse_args()
    return args


def compose_url(
    page: int, api_key: str, text: str, min_date: float, max_date: float
) -> str:
    query = urllib.parse.urlencode(
        {
            "method": "flickr.photos.search",
            "api_key": api_key,
            "text": text,
            "sort": SORT,
            "per_page": PER_PAGE,
            "page": page,
            "min_upload_date": min_date,
            "max_upload_date": max_date,
            "format": "rest",
            "extras": "url_o,url_l,url_c,url_z,url_n",
        }
    )
    return f"{SEARCH_URL}?{query}"


def parse_page(
    page: int, api_key: str, text: str, min_date: float, max_date: float
) -> tuple[dict[str, str], tuple[dict[str, str], ...]]:
    for _ in range(MAX_PAGE_REQUESTS):
        try:
            with urllib.request.urlopen(
                compose_url(page, api_key, text, min_date, max_date),
                timeout=MAX_PAGE_TIMEOUT,
            ) as response:
                root = ElementTree.parse(response).getroot()
        except TimeoutError:
            continue
        else:
            break

    else:
        return {
            "pages": "0",
            "total": "0",
            "page": "0",
            "perpage": "0",
        }, tuple()

    if root.attrib["stat"] != "ok":
        error = root.find("err")
        if error is not None:
            raise OSError(
                f"Flickr error {error.get('code')}: {error.get('msg')}"
            )
        raise OSError("Flickr search failed")

    metadata = root.find("photos")
    assert metadata is not None

    photos = tuple(photo.attrib for photo in root.iter("photo"))

    return metadata.attrib, photos


class PhotoDownloader:
    def __init__(self, image_path: str) -> None:
        self.image_path = Path(image_path)
        self.image_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, photo: dict[str, str]) -> None:
        # Find the URL corresponding to the highest image resolution. We will
        # need this URL here to determine the image extension (typically .jpg,
        # but could be .png, .gif, etc).
        url: str | None = None
        for url_suffix in ("o", "l", "k", "h", "b", "c", "z", "n"):
            url_attr = f"url_{url_suffix}"
            if photo.get(url_attr):
                url = photo[url_attr]
                break

        if url is not None:
            url_filename = urllib.parse.urlparse(url).path
            image_ext = os.path.splitext(url_filename)[1]

            image_name = f"{photo['id']}_{photo['secret']}{image_ext}"
            path = self.image_path / image_name
            if path.exists():
                return
            print(url)
            with tempfile.TemporaryDirectory(dir=self.image_path) as directory:
                partial = Path(directory) / path.name
                for _ in range(MAX_IMAGE_REQUESTS):
                    try:
                        urllib.request.urlretrieve(url, str(partial))
                    except urllib.error.ContentTooShortError:
                        continue
                    partial.replace(path)
                    return
                print(f"Skipping incomplete photo: {image_name}")


def main() -> None:
    args = parse_args()

    downloader = PhotoDownloader(args.image_path)
    pool = multiprocessing.Pool(processes=args.num_procs)

    num_pages = float("inf")
    page = 0

    min_date = MIN_DATE
    max_date = MAX_DATE

    days_in_row = 0

    while num_pages > page:
        page += 1

        metadata, photos = parse_page(
            page, args.api_key, args.search_text, min_date, max_date
        )

        num_pages = int(metadata["pages"])

        print(78 * "=")
        print("Page:\t\t", page, "of", num_pages)
        print("Min-Date:\t", datetime.datetime.fromtimestamp(min_date))
        print("Max-Date:\t", datetime.datetime.fromtimestamp(max_date))
        print("Num-Photos:\t", len(photos))
        print(78 * "=")

        try:
            pool.map_async(downloader, photos).get(1e10)
        except KeyboardInterrupt:
            pool.close()
            pool.join()
            break

        if page >= num_pages:
            max_date -= TIME_SKIP
            min_date -= TIME_SKIP
            page = 0

        if num_pages == 0:
            days_in_row = days_in_row + 1
            num_pages = float("inf")

            print("    No images in", days_in_row, "days in a row")

            if days_in_row == args.max_days_without_image:
                break
        else:
            days_in_row = 0


if __name__ == "__main__":
    main()
