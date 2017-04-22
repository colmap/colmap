# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

# This script exports a COLMAP database to the file structure to run VisualSfM.

import os
import sys
import argparse
import sqlite3
import shutil
import gzip
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--min_num_matches", type=int, default=15)
    args = parser.parse_args()
    return args


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def main():
    args = parse_args()

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    try:
        os.makedirs(args.output_path)
    except:
        pass

    cameras = {}
    cursor.execute("SELECT camera_id, params FROM cameras;")
    for row in cursor:
        camera_id = row[0]
        params = np.fromstring(row[1], dtype=np.double)
        cameras[camera_id] = params

    images = {}
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        camera_id = row[1]
        image_name = row[2]
        print "Copying image", image_name
        images[image_id] = (len(images), image_name)
        if not os.path.exists(os.path.join(args.output_path, image_name)):
            shutil.copyfile(os.path.join(args.image_path, image_name),
                            os.path.join(args.output_path, image_name))

    for image_id, (image_idx, image_name) in images.iteritems():
        print "Exporting key file for", image_name
        base_name, ext = os.path.splitext(image_name)
        key_file_name = os.path.join(args.output_path, base_name + ".sift")
        if os.path.exists(key_file_name):
            continue

        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;",
                       (image_id,))
        row = next(cursor)
        if row[0] is None:
            keypoints = np.zeros((0, 4), dtype=np.float32)
            descriptors = np.zeros((0, 128), dtype=np.uint8)
        else:
            keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 4)
            cursor.execute("SELECT data FROM descriptors WHERE image_id=?;",
                           (image_id,))
            row = next(cursor)
            descriptors = np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128)

        with open(key_file_name, "w") as fid:
            fid.write("%d %d\n" % (keypoints.shape[0], descriptors.shape[1]))
            for r in range(keypoints.shape[0]):
                fid.write("%f %f %f %f " % (keypoints[r, 0], keypoints[r, 1],
                                            keypoints[r, 2], keypoints[r, 3]))
                fid.write(" ".join(map(str, descriptors[r].ravel().tolist())))
                fid.write("\n")

    with open(os.path.join(args.output_path, "matches.txt"), "w") as fid:
        cursor.execute("SELECT pair_id, data FROM inlier_matches "
                       "WHERE rows>=?;", (args.min_num_matches,))
        for row in cursor:
            pair_id = row[0]
            inlier_matches = np.fromstring(row[1],
                                           dtype=np.uint32).reshape(-1, 2)
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_name1 = images[image_id1][1]
            image_name2 = images[image_id2][1]
            fid.write("%s %s %d\n" % (image_name1, image_name2,
                                      inlier_matches.shape[0]))
            line1 = ""
            line2 = ""
            for i in range(inlier_matches.shape[0]):
                line1 += "%d " % inlier_matches[i, 0]
                line2 += "%d " % inlier_matches[i, 1]
            fid.write(line1 + "\n")
            fid.write(line2 + "\n")

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
