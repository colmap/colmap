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

# This script exports inlier image pairs from a COLMAP database to a text file.

import sqlite3
import argparse


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
    cursor.execute('SELECT image_id, name FROM images;')
    for row in cursor:
        image_id = row[0]
        name = row[1]
        image_id_to_name[image_id] = name

    # Iterate over entries in the inlier_matches table
    output = open(args.match_list_path, 'w')
    cursor.execute('SELECT pair_id, rows FROM inlier_matches;')
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
