# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
#                     Filip Radenovic <filip.radenovic at cmp.felk.cvut.cz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This script downloads images from Flickr for a given search query, ordered
# from most recent to older images. If the downloader does not find images for
# more than `max_days_without_image` in a row, it exits to avoid an infinite
# loop. Note that too many parallel download processes may lead to being banned
# from Flickr for a certain time.

import os
import time
import datetime
import urllib
import urllib2
import socket
import argparse
import multiprocessing
import xml.etree.ElementTree as ElementTree


PER_PAGE = 500
SORT = "date-posted-desc"
URL = "https://api.flickr.com/services/rest/?method=flickr.photos.search&" \
      "api_key=%s&text=%s&sort=%s&per_page=%d&page=%d&min_upload_date=%s&" \
      "max_upload_date=%s&format=rest&extras=url_o,url_l,url_c,url_z,url_n"
MAX_PAGE_REQUESTS = 5
MAX_PAGE_TIMEOUT = 20
MAX_IMAGE_REQUESTS = 3
TIME_SKIP = 24 * 60 * 60
MAX_DATE = time.time()
MIN_DATE = MAX_DATE - TIME_SKIP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_text", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--num_procs", type=int, default=10)
    parser.add_argument("--max_days_without_image", type=int, default=365)
    args = parser.parse_args()
    return args


def compose_url(page, api_key, text, min_date, max_date):
    return URL % (api_key, text, SORT, PER_PAGE, page,
                  str(min_date), str(max_date))


def parse_page(page, api_key, text, min_date, max_date):
    f = None
    for _ in range(MAX_PAGE_REQUESTS):
        try:
            f = urllib2.urlopen(compose_url(page, api_key, text, min_date,
                                            max_date), timeout=MAX_PAGE_TIMEOUT)
        except socket.timeout:
            continue
        else:
            break

    if f is None:
        return {'pages': '0',
                'total': '0',
                'page': '0',
                'perpage': '0'}, tuple()

    response = f.read()
    root = ElementTree.fromstring(response)

    if root.attrib["stat"] != "ok":
        raise IOError

    photos = []
    for photo in root.iter("photo"):
        photos.append(photo.attrib)

    return root.find("photos").attrib, photos


class PhotoDownloader(object):

    def __init__(self, image_path):
        self.image_path = image_path

    def __call__(self, photo):
        image_name = "%s_%s.jpg" % (photo["id"], photo["secret"])
        path = os.path.join(self.image_path, image_name)
        if not os.path.exists(path):
            url = None
            for url_suffix in ("o", "l", "k", "h", "b", "c", "z"):
                url_attr = "url_%s" % url_suffix
                if photo.get(url_attr) is not None:
                    url = photo.get(url_attr)
                    break
            if url is not None:
                print url
                for _ in range(MAX_IMAGE_REQUESTS):
                    try:
                        urllib.urlretrieve(url, path)
                    except urllib.ContentTooShortError:
                        continue
                    else:
                        break


def main():
    args = parse_args()

    downloader = PhotoDownloader(args.image_path)
    pool = multiprocessing.Pool(processes=args.num_procs)

    num_pages = float("inf")
    page = 0

    min_date = MIN_DATE
    max_date = MAX_DATE

    days_in_row = 0;

    search_text = args.search_text.replace(" ", "-")

    while num_pages > page:
        page += 1

        metadata, photos = parse_page(page, args.api_key, search_text,
                                      min_date, max_date)

        num_pages = int(metadata["pages"])

        print 78 * "="
        print "Page:\t\t", page, "of", num_pages
        print "Min-Date:\t", datetime.datetime.fromtimestamp(min_date)
        print "Max-Date:\t", datetime.datetime.fromtimestamp(max_date)
        print "Num-Photos:\t", len(photos)
        print 78 * "="

        try:
            pool.map_async(downloader, photos).get(1e10)
        except KeyboardInterrupt:
            pool.wait()
            break

        if page >= num_pages:
            max_date -= TIME_SKIP
            min_date -= TIME_SKIP
            page = 0

        if num_pages == 0:
            days_in_row = days_in_row + 1
            num_pages = float("inf")

            print "    No images in", days_in_row, "days in a row"

            if days_in_row == args.max_days_without_image:
                break
        else:
            days_in_row = 0


if __name__ == "__main__":
    main()
