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
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import os
import collections
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

from read_write_model import read_next_bytes, write_next_bytes


MeshPoint = collections.namedtuple(
    "MeshingPoint", ["position", "color", "normal", "num_visible_images", "visible_image_idxs"])


def read_fused(path_to_fused_ply, path_to_fused_ply_vis):
    """
    see: src/mvs/meshing.cc
        void ReadDenseReconstruction(const std::string& path
    """
    assert os.path.isfile(path_to_fused_ply)
    assert os.path.isfile(path_to_fused_ply_vis)

    point_cloud = PyntCloud.from_file(path_to_fused_ply)
    xyz_arr = point_cloud.points.loc[:, ["x", "y", "z"]].to_numpy()
    normal_arr = point_cloud.points.loc[:, ["nx", "ny", "nz"]].to_numpy()
    color_arr = point_cloud.points.loc[:, ["red", "green", "blue"]].to_numpy()

    with open(path_to_fused_ply_vis, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        mesh_points = [0] * num_points
        for i in range(num_points):
            num_visible_images = read_next_bytes(fid, 4, "I")[0]
            visible_image_idxs = read_next_bytes(
                fid, num_bytes=4*num_visible_images,
                format_char_sequence="I"*num_visible_images)
            visible_image_idxs = np.array(tuple(map(int, visible_image_idxs)))
            mesh_point = MeshPoint(
                position=xyz_arr[i],
                color=color_arr[i],
                normal=normal_arr[i],
                num_visible_images=num_visible_images,
                visible_image_idxs=visible_image_idxs)
            mesh_points[i] = mesh_point
        return mesh_points


def write_fused_ply(mesh_points, path_to_fused_ply):
    columns = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"]
    points_data_frame = pd.DataFrame(
        np.zeros((len(mesh_points), len(columns))),
        columns=columns)

    positions = np.asarray([point.position for point in mesh_points])
    normals = np.asarray([point.normal for point in mesh_points])
    colors = np.asarray([point.color for point in mesh_points])

    points_data_frame.loc[:, ["x", "y", "z"]] = positions
    points_data_frame.loc[:, ["nx", "ny", "nz"]] = normals
    points_data_frame.loc[:, ["red", "green", "blue"]] = colors

    points_data_frame = points_data_frame.astype({
        "x": positions.dtype, "y": positions.dtype, "z": positions.dtype,
        "red": colors.dtype, "green": colors.dtype, "blue": colors.dtype,
        "nx": normals.dtype, "ny": normals.dtype, "nz": normals.dtype})

    point_cloud = PyntCloud(points_data_frame)
    point_cloud.to_file(path_to_fused_ply)


def write_fused_ply_vis(mesh_points, path_to_fused_ply_vis):
    """
    see: src/mvs/fusion.cc
        void WritePointsVisibility(const std::string& path, const std::vector<std::vector<int>>& points_visibility)
    """
    with open(path_to_fused_ply_vis, "wb") as fid:
        write_next_bytes(fid, len(mesh_points), "Q")
        for point in mesh_points:
            write_next_bytes(fid, point.num_visible_images, "I")
            format_char_sequence = "I"*point.num_visible_images
            write_next_bytes(fid, [*point.visible_image_idxs], format_char_sequence)


def write_fused(points, path_to_fused_ply, path_to_fused_ply_vis):
    write_fused_ply(points, path_to_fused_ply)
    write_fused_ply_vis(points, path_to_fused_ply_vis)
