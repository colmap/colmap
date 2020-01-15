#!/usr/bin/env python

# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import argparse
import numpy as np
import os
import pylab as plt


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--depth_map",
                        help="path to depth map", type=str, required=True)
    parser.add_argument("-n", "--normal_map",
                        help="path to normal map", type=str, required=True)
    parser.add_argument("--min_depth", help="minimum visualization depth",
                        type=float, default=5)
    parser.add_argument("--max_depth", help="maximum visualization depth",
                        type=float, default=95)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.min_depth > args.max_depth:
        raise ValueError(
            "Minimum depth should be less than or equal to the maximum depth.")

    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists(args.depth_map):
        raise FileNotFoundError("File not found: {}".format(args.depth_map))

    if not os.path.exists(args.normal_map):
        raise FileNotFoundError("File not found: {}".format(args.normal_map))

    depth_map = read_array(args.depth_map)
    normal_map = read_array(args.normal_map)

    # Visualize the depth map.
    min_depth, max_depth = np.percentile(
        depth_map, [args.min_depth, args.max_depth])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    plt.imshow(depth_map)
    plt.show()


if __name__ == "__main__":
    main()
