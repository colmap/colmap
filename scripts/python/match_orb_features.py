# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

# This script exports inlier matches from a COLMAP database to a text file.

import os
import argparse
import sqlite3
import numpy as np
import cv2
import glob
import pylab as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    args = parser.parse_args()
    return args


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def main():
    args = parse_args()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    cursor.execute("SELECT image_id, data FROM descriptors;")
    descriptors = [(row[0], np.fromstring(row[1], dtype=np.uint8).reshape(-1, 32))
                   for row in cursor]

    for i in range(len(descriptors)):
        image_id1, descriptors1 = descriptors[i]
        for j in range(i):
            image_id2, descriptors2 = descriptors[j]

            print image_id1, image_id2

            pair_id = image_ids_to_pair_id(image_id1, image_id2)

            match_list = bf.knnMatch(descriptors1, descriptors2, k=2)

            match_list_unambiguous = []
            for m1, m2 in match_list:
                if m1.distance < 0.70 * m2.distance:
                    match_list_unambiguous.append([m1])

            match_list = bf.match(descriptors1, descriptors2)

            matches = np.zeros((len(match_list_unambiguous), 2), dtype=np.uint32)
            for k in range(matches.shape[0]):
                matches[k, 0] = match_list_unambiguous[k].trainIdx
                matches[k, 1] = match_list_unambiguous[k].queryIdx

            matches = np.ascontiguousarray(matches)
            cursor.execute("INSERT INTO matches (rows, cols, data, pair_id) VALUES (?, 2, ?, ?);",
                           (matches.shape[0], np.getbuffer(matches), pair_id))
            connection.commit()

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
