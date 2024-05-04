#!/usr/bin/env python

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


import argparse
import numpy as np
import os
import struct


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
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


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list
        )
        fid.write(byte_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--depth_map", help="path to depth map", type=str, required=True
    )
    parser.add_argument(
        "-n", "--normal_map", help="path to normal map", type=str, required=True
    )
    parser.add_argument(
        "--min_depth_percentile",
        help="minimum visualization depth percentile",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--max_depth_percentile",
        help="maximum visualization depth percentile",
        type=float,
        default=95,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.min_depth_percentile > args.max_depth_percentile:
        raise ValueError(
            "min_depth_percentile should be less than or equal "
            "to the max_depth_percentile."
        )

    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists(args.depth_map):
        raise FileNotFoundError("File not found: {}".format(args.depth_map))

    if not os.path.exists(args.normal_map):
        raise FileNotFoundError("File not found: {}".format(args.normal_map))

    depth_map = read_array(args.depth_map)
    normal_map = read_array(args.normal_map)

    min_depth, max_depth = np.percentile(
        depth_map, [args.min_depth_percentile, args.max_depth_percentile]
    )
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    import pylab as plt

    # Visualize the depth map.
    plt.figure()
    plt.imshow(depth_map)
    plt.title("depth map")

    # Visualize the normal map.
    plt.figure()
    plt.imshow(normal_map)
    plt.title("normal map")

    plt.show()


if __name__ == "__main__":
    main()
