# Copyright (c), ETH Zurich and UNC Chapel Hill.
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


# This script converts a Bundler reconstruction file to a PLY point cloud.

import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundler_path", required=True)
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--normalize_p0", type=float, default=0.2)
    parser.add_argument("--normalize_p1", type=float, default=0.8)
    parser.add_argument("--min_track_length", type=int, default=3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.bundler_path, "r") as fid:
        line = fid.readline()
        line = fid.readline()
        num_images, num_points = map(int, line.split())

        for i in range(5 * num_images):
            fid.readline()

        xyz = np.zeros((num_points, 3), dtype=np.float64)
        rgb = np.zeros((num_points, 3), dtype=np.uint16)
        track_lengths = np.zeros((num_points,), dtype=np.uint32)

        for i in range(num_points):
            if i % 1000 == 0:
                print("Reading point", i, "/", num_points)
            xyz[i] = map(float, fid.readline().split())
            rgb[i] = map(int, fid.readline().split())
            track_lengths[i] = int(fid.readline().split()[0])

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

        bbox_min = np.array(
            [sorted_x[min_coord], sorted_y[min_coord], sorted_z[min_coord]]
        )
        bbox_max = np.array(
            [sorted_x[max_coord], sorted_y[max_coord], sorted_z[max_coord]]
        )

        extent = np.linalg.norm(bbox_max - bbox_min)
        scale = 10.0 / extent

        xyz -= mean_coords
        xyz *= scale

    xyz[:, 2] *= -1

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
                print("Writing point", i, "/", xyz.shape[0])
            fid.write(
                "%f %f %f 0 0 0 %d %d %d\n"
                % (
                    xyz[i, 0],
                    xyz[i, 1],
                    xyz[i, 2],
                    rgb[i, 0],
                    rgb[i, 1],
                    rgb[i, 2],
                )
            )


if __name__ == "__main__":
    main()
