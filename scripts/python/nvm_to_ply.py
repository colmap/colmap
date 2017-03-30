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

# This script converts a VisualSfM reconstruction file to a PLY point cloud.

import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvm_path", required=True)
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--normalize_p0", type=float, default=0.2)
    parser.add_argument("--normalize_p1", type=float, default=0.8)
    parser.add_argument("--min_track_length", type=int, default=3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.nvm_path, "r") as fid:
        line = fid.readline()
        line = fid.readline()
        num_images = int(fid.readline())

        for i in range(num_images + 1):
            fid.readline()

        num_points = int(fid.readline())

        xyz = np.zeros((num_points, 3), dtype=np.float64)
        rgb = np.zeros((num_points, 3), dtype=np.uint16)
        track_lengths = np.zeros((num_points,), dtype=np.uint32)

        for i in range(num_points):
            if i % 1000 == 0:
                print "Reading point", i, "/", num_points
            elems = fid.readline().split()
            xyz[i] = map(float, elems[0:3])
            rgb[i] = map(int, elems[3:6])
            track_lengths[i] = int(elems[6])

    mask = track_lengths >= args.min_track_length
    xyz = xyz[mask]
    rgb = rgb[mask]

    if args.normalize:
        sorted_x = np.sort(xyz[:, 0])
        sorted_y = np.sort(xyz[:, 1])
        sorted_z = np.sort(xyz[:, 2])

        num_coords = sorted_x.size
        min_coord = int(args.normalize_p0 * num_coords)
        max_coord = int(args.normalize_p1 * num_coords)
        mean_coords = xyz.mean(0)

        bbox_min = np.array([sorted_x[min_coord], sorted_y[min_coord],
                             sorted_z[min_coord]])
        bbox_max = np.array([sorted_x[max_coord], sorted_y[max_coord],
                             sorted_z[max_coord]])

        extent = np.linalg.norm(bbox_max - bbox_min)
        scale = 10.0 / extent

        xyz -= mean_coords
        xyz *= scale

    with open(args.ply_path, "w") as fid:
        fid.write("ply\n")
        fid.write("format ascii 1.0\n")
        fid.write("element vertex %d\n" % xyz.shape[0])
        fid.write("property float x\n")
        fid.write("property float y\n")
        fid.write("property float z\n")
        fid.write("property float nx\n")
        fid.write("property float ny\n")
        fid.write("property float nz\n")
        fid.write("property uchar diffuse_red\n")
        fid.write("property uchar diffuse_green\n")
        fid.write("property uchar diffuse_blue\n")
        fid.write("end_header\n")
        for i in range(xyz.shape[0]):
            if i % 1000 == 0:
                print "Writing point", i, "/", xyz.shape[0]
            fid.write("%f %f %f 0 0 0 %d %d %d\n" % (xyz[i, 0], xyz[i, 1],
                                                     xyz[i, 2], rgb[i, 0],
                                                     rgb[i, 1], rgb[i, 2]))


if __name__ == "__main__":
    main()
