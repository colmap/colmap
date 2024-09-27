# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# This script exports a COLMAP database to the file structure to run VisualSfM.

import argparse
import os
import shutil
import sqlite3
import struct

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--min_num_matches", type=int, default=15)
    parser.add_argument("--binary_feature_files", type=bool, default=True)
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
    except:  # noqa E722
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
        print("Copying image", image_name)
        images[image_id] = (len(images), image_name)
        if not os.path.exists(os.path.join(args.output_path, image_name)):
            shutil.copyfile(
                os.path.join(args.image_path, image_name),
                os.path.join(args.output_path, image_name),
            )

    # The magic numbers used in VisualSfM's binary file format for storing the
    # feature descriptors.
    sift_name = 1413892435
    sift_version_v4 = 808334422
    sift_eof_marker = 1179600383

    for image_id, (image_idx, image_name) in images.iteritems():
        print("Exporting key file for", image_name)
        base_name, ext = os.path.splitext(image_name)
        key_file_name = os.path.join(args.output_path, base_name + ".sift")
        if os.path.exists(key_file_name):
            continue

        cursor.execute(
            "SELECT data FROM keypoints WHERE image_id=?;", (image_id,)
        )
        row = next(cursor)
        if row[0] is None:
            keypoints = np.zeros((0, 6), dtype=np.float32)
            descriptors = np.zeros((0, 128), dtype=np.uint8)
        else:
            keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 6)
            cursor.execute(
                "SELECT data FROM descriptors WHERE image_id=?;", (image_id,)
            )
            row = next(cursor)
            descriptors = np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128)

        if args.binary_feature_files:
            with open(key_file_name, "wb") as fid:
                fid.write(struct.pack("i", sift_name))
                fid.write(struct.pack("i", sift_version_v4))
                fid.write(struct.pack("i", keypoints.shape[0]))
                fid.write(struct.pack("i", 4))
                fid.write(struct.pack("i", 128))
                keypoints[:, :4].astype(np.float32).tofile(fid)
                descriptors.astype(np.uint8).tofile(fid)
                fid.write(struct.pack("i", sift_eof_marker))
        else:
            with open(key_file_name, "w") as fid:
                fid.write(
                    "%d %d\n" % (keypoints.shape[0], descriptors.shape[1])
                )
                for r in range(keypoints.shape[0]):
                    fid.write("%f %f 0 0 " % (keypoints[r, 0], keypoints[r, 1]))
                    fid.write(
                        " ".join(map(str, descriptors[r].ravel().tolist()))
                    )
                    fid.write("\n")

    with open(os.path.join(args.output_path, "matches.txt"), "w") as fid:
        cursor.execute(
            "SELECT pair_id, data FROM two_view_geometries " "WHERE rows>=?;",
            (args.min_num_matches,),
        )
        for row in cursor:
            pair_id = row[0]
            inlier_matches = np.fromstring(row[1], dtype=np.uint32).reshape(
                -1, 2
            )
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_name1 = images[image_id1][1]
            image_name2 = images[image_id2][1]
            fid.write(
                "%s %s %d\n"
                % (image_name1, image_name2, inlier_matches.shape[0])
            )
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
