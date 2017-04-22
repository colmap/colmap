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

# This script exports a COLMAP database to the file structure to run Bundler.

import os
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
    with open(os.path.join(args.output_path, "list.txt"), "w") as fid:
        cursor.execute("SELECT image_id, camera_id, name FROM images;")
        for row in cursor:
            image_id = row[0]
            camera_id = row[1]
            image_name = row[2]
            print "Copying image", image_name
            images[image_id] = (len(images), image_name)
            fid.write("./%s 0 %f\n" % (image_name, cameras[camera_id][0]))
            if not os.path.exists(os.path.join(args.output_path, image_name)):
                shutil.copyfile(os.path.join(args.image_path, image_name),
                                os.path.join(args.output_path, image_name))

    for image_id, (image_idx, image_name) in images.iteritems():
        print "Exporting key file for", image_name
        base_name, ext = os.path.splitext(image_name)
        key_file_name = os.path.join(args.output_path, base_name + ".key")
        key_file_name_gz = key_file_name + ".gz"
        if os.path.exists(key_file_name_gz):
            continue

        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;",
                       (image_id,))
        row = next(cursor)
        keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 4)
        cursor.execute("SELECT data FROM descriptors WHERE image_id=?;",
                       (image_id,))
        row = next(cursor)
        descriptors = np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128)

        with open(key_file_name, "w") as fid:
            fid.write("%d %d\n" % (keypoints.shape[0], descriptors.shape[1]))
            for r in range(keypoints.shape[0]):
                fid.write("%f %f %f %f\n" % (keypoints[r, 1], keypoints[r, 0],
                                             keypoints[r, 2], keypoints[r, 3]))
                for i in range(0, 128, 20):
                    desc_block = descriptors[r, i:i+20]
                    fid.write(" ".join(map(str, desc_block.ravel().tolist())))
                    fid.write("\n")

        with open(key_file_name, "rb") as fid_in:
            with gzip.open(key_file_name + ".gz", "wb") as fid_out:
                fid_out.writelines(fid_in)

        os.remove(key_file_name)

    with open(os.path.join(args.output_path, "matches.init.txt"), "w") as fid:
        cursor.execute("SELECT pair_id, data FROM inlier_matches "
                       "WHERE rows>=?;", (args.min_num_matches,))
        for row in cursor:
            pair_id = row[0]
            inlier_matches = np.fromstring(row[1],
                                           dtype=np.uint32).reshape(-1, 2)
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_idx1 = images[image_id1][0]
            image_idx2 = images[image_id2][0]
            fid.write("%d %d\n%d\n" % (image_idx1, image_idx2,
                                       inlier_matches.shape[0]))
            for i in range(inlier_matches.shape[0]):
                fid.write("%d %d\n" % (inlier_matches[i, 0],
                                       inlier_matches[i, 1]))

    with open(os.path.join(args.output_path, "run_bundler.sh"), "w") as fid:
        fid.write("bin/Bundler list.txt \\\n")
        fid.write("--run_bundle \\\n")
        fid.write("--use_focal_estimate \\\n")
        fid.write("--output_all bundle_ \\\n")
        fid.write("--constrain_focal \\\n")
        fid.write("--estimate_distortion \\\n")
        fid.write("--match_table matches.init.txt \\\n")
        fid.write("--variable_focal_length \\\n")
        fid.write("--output_dir bundle \\\n")
        fid.write("--output bundle.out \\\n")
        fid.write("--constrain_focal_weight 0.0001 \\\n")

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
