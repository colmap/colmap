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

set(FOLDER_NAME "base")

COLMAP_ADD_SOURCES(
    camera.h camera.cc
    camera_database.h camera_database.cc
    camera_models.h camera_models.cc
    camera_rig.h camera_rig.cc
    correspondence_graph.h correspondence_graph.cc
    database.h database.cc
    database_cache.h database_cache.cc
    essential_matrix.h essential_matrix.cc
    gps.h gps.cc
    graph_cut.h graph_cut.cc
    homography_matrix.h homography_matrix.cc
    image.h image.cc
    image_reader.h image_reader.cc
    line.h line.cc
    point2d.h point2d.cc
    point3d.h point3d.cc
    polynomial.h polynomial.cc
    pose.h pose.cc
    projection.h projection.cc
    reconstruction.h reconstruction.cc
    reconstruction_manager.h reconstruction_manager.cc
    scene_clustering.h scene_clustering.cc
    similarity_transform.h similarity_transform.cc
    track.h track.cc
    triangulation.h triangulation.cc
    undistortion.h undistortion.cc
    visibility_pyramid.h visibility_pyramid.cc
    warp.h warp.cc
)

COLMAP_ADD_TEST(camera_database_test camera_database_test.cc)
COLMAP_ADD_TEST(camera_models_test camera_models_test.cc)
COLMAP_ADD_TEST(camera_rig_test camera_rig_test.cc)
COLMAP_ADD_TEST(camera_test camera_test.cc)
COLMAP_ADD_TEST(correspondence_graph_test correspondence_graph_test.cc)
COLMAP_ADD_TEST(cost_functions_test cost_functions_test.cc)
COLMAP_ADD_TEST(database_cache_test database_cache_test.cc)
COLMAP_ADD_TEST(database_test database_test.cc)
COLMAP_ADD_TEST(essential_matrix_utils_test essential_matrix_test.cc)
COLMAP_ADD_TEST(gps_test gps_test.cc)
COLMAP_ADD_TEST(graph_cut_test graph_cut_test.cc)
COLMAP_ADD_TEST(homography_matrix_utils_test homography_matrix_test.cc)
COLMAP_ADD_TEST(image_test image_test.cc)
COLMAP_ADD_TEST(line_test line_test.cc)
COLMAP_ADD_TEST(point2d_test point2d_test.cc)
COLMAP_ADD_TEST(point3d_test point3d_test.cc)
COLMAP_ADD_TEST(polynomial_test polynomial_test.cc)
COLMAP_ADD_TEST(pose_test pose_test.cc)
COLMAP_ADD_TEST(projection_test projection_test.cc)
COLMAP_ADD_TEST(reconstruction_test reconstruction_test.cc)
COLMAP_ADD_TEST(reconstruction_manager_test reconstruction_manager_test.cc)
COLMAP_ADD_TEST(scene_clustering_test scene_clustering_test.cc)
COLMAP_ADD_TEST(similarity_transform_test similarity_transform_test.cc)
COLMAP_ADD_TEST(track_test track_test.cc)
COLMAP_ADD_TEST(triangulation_test triangulation_test.cc)
COLMAP_ADD_TEST(undistortion_test undistortion_test.cc)
COLMAP_ADD_TEST(visibility_pyramid_test visibility_pyramid_test.cc)
COLMAP_ADD_TEST(warp_test warp_test.cc)
