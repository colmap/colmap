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

# This script merges multiple homogeneous PLY files into a single PLY file.

import os
import glob
import argparse
import numpy as np
import plyfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True)
    parser.add_argument("--merged_path", required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    files = []
    for file_name in os.listdir(args.folder_path):
        if len(file_name) < 4 or file_name[-4:].lower() != ".ply":
            continue

        print "Reading file", file_name
        file = plyfile.PlyData.read(os.path.join(args.folder_path, file_name))
        for element in file.elements:
            files.append(element.data)

    print "Merging files"
    merged_file = np.concatenate(files, -1)
    merged_el = plyfile.PlyElement.describe(merged_file, 'vertex')

    print "Writing merged file"
    plyfile.PlyData([merged_el]).write(args.merged_path)


if __name__ == '__main__':
    main()
