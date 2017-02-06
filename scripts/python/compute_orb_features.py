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
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--num_features", type=int, default=4096)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    cursor.execute("SELECT image_id, name FROM images;")
    images = [(row[0], row[1]) for row in cursor]

    for image_id, image_name in images:
        print image_name

        image = cv2.imread(os.path.join(args.image_path, image_name), 0)

        keypoint_list = surf.detect(image, None)
        keypoint_list = keypoint_list[len(keypoint_list) -
                                      min(args.num_features,
                                          len(keypoint_list)):]
        keypoint_list, descriptors = orb.compute(image, keypoint_list)

        keypoints = np.zeros((len(keypoint_list), 4), dtype=np.float32)
        for i in range(keypoints.shape[0]):
            keypoints[i, 0] = keypoint_list[i].pt[0]
            keypoints[i, 1] = keypoint_list[i].pt[1]
            keypoints[i, 2] = keypoint_list[i].size
            keypoints[i, 3] = keypoint_list[i].angle

        keypoints = np.ascontiguousarray(keypoints)
        cursor.execute("UPDATE keypoints SET rows=?, cols=4, data=? WHERE image_id=?",
                       (keypoints.shape[0], np.getbuffer(keypoints), image_id))

        descriptors = np.ascontiguousarray(descriptors)
        cursor.execute("UPDATE descriptors SET rows=?, cols=32, data=? WHERE image_id=?",
                       (descriptors.shape[0], np.getbuffer(descriptors), image_id))

        connection.commit()

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
