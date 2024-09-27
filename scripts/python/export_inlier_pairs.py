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


# This script exports inlier image pairs from a COLMAP database to a text file.

import argparse
import sqlite3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--match_list_path", required=True)
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

    # Get a mapping between image ids and image names
    image_id_to_name = dict()
    cursor.execute("SELECT image_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        name = row[1]
        image_id_to_name[image_id] = name

    # Iterate over entries in the two_view_geometries table
    output = open(args.match_list_path, "w")
    cursor.execute("SELECT pair_id, rows FROM two_view_geometries;")
    for row in cursor:
        pair_id = row[0]
        rows = row[1]

        if rows < args.min_num_matches:
            continue

        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        image_name1 = image_id_to_name[image_id1]
        image_name2 = image_id_to_name[image_id2]

        output.write("%s %s\n" % (image_name1, image_name2))

    output.close()
    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
