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

# This script exports a COLMAP database to the file structure to run Bundler.

import os
import shutil
import argparse
import sqlite3
import collections

import numpy as np
import pylab as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import spectral_clustering


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--image_path")
    parser.add_argument("--output_path")
    parser.add_argument("--min_num_matches", type=int, default=15)
    # parser.add_argument("--max_num_matches", type=float, default=200)
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

    cursor.execute("SELECT pair_id, rows FROM inlier_matches "
                   "WHERE rows>=?;", (args.min_num_matches,))

    image_ids = set()
    scene_graph = collections.defaultdict(list)
    for row in cursor:
        image_id1, image_id2 = pair_id_to_image_ids(row[0])
        image_ids.add(image_id1)
        image_ids.add(image_id2)
        num_matches = row[1]
        scene_graph[image_id1].append((image_id2, num_matches))
        scene_graph[image_id2].append((image_id1, num_matches))

    image_ids = tuple(image_ids)
    image_names = []
    for image_id in image_ids:
        cursor.execute("SELECT name FROM images WHERE image_id=?;", (image_id,))
        image_names.append(next(cursor)[0])

    match_matrix = np.zeros((len(image_ids), len(image_ids)), dtype=np.float32)

    for i, image_id1 in enumerate(image_ids):
        match_matrix[i, i] = 0
        for image_id2, num_matches in scene_graph[image_id1]:
            j = image_ids.index(image_id2)
            match_matrix[i, j] = num_matches

    plt.clf()
    plt.imshow(match_matrix, interpolation="nearest")
    plt.savefig("/Users/jsch/Desktop/match_matrix.png", dpi=300)

    plt.clf()
    plt.hist(match_matrix.ravel(), 50)
    plt.savefig("/Users/jsch/Desktop/match_matrix_hist.png", dpi=300)

    labels = DBSCAN(eps=500, min_samples=10).fit(match_matrix).labels_
    labels = spectral_clustering(match_matrix, 2)

    print(labels)

    np.savetxt("/Users/jsch/Desktop/labels.txt", labels)
    np.savetxt("/Users/jsch/Desktop/image_ids.txt", np.array(image_ids))

    for label in np.unique(labels):
        print("LABEL:", label)
        for i in range(len(image_ids)):
            if labels[i] == label:
                print(" ", image_names[i])

    cursor.close()
    connection.close()

    if args.output_path and args.image_path:
        for label in np.unique(labels):
            cluster_path = os.path.join(args.output_path, str(label))
            if not os.path.exists(cluster_path):
                os.makedirs(cluster_path)
            for i in range(len(image_ids)):
                if labels[i] == label:
                    image_name = image_names[i]
                    shutil.copyfile(os.path.join(args.image_path, image_name),
                                    os.path.join(cluster_path, image_name))


if __name__ == "__main__":
    main()
