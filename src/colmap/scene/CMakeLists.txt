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


set(FOLDER_NAME "scene")

COLMAP_ADD_LIBRARY(
    NAME colmap_scene
    SRCS
        camera.h camera.cc
        correspondence_graph.h correspondence_graph.cc
        database.h database.cc
        database_cache.h database_cache.cc
        frame.h frame.cc
        image.h image.cc
        point2d.h point2d.cc
        point3d.h point3d.cc
        projection.h projection.cc
        reconstruction.h reconstruction.cc
        reconstruction_io.h reconstruction_io.cc
        reconstruction_io_binary.h reconstruction_io_binary.cc
        reconstruction_io_text.h reconstruction_io_text.cc
        reconstruction_io_utils.h reconstruction_io_utils.cc
        reconstruction_manager.h reconstruction_manager.cc
        rig.h rig.cc
        scene_clustering.h scene_clustering.cc
        synthetic.h synthetic.cc
        track.h track.cc
        two_view_geometry.h two_view_geometry.cc
        visibility_pyramid.h visibility_pyramid.cc
    PUBLIC_LINK_LIBS
        colmap_sensor
        colmap_feature_types
        colmap_geometry
        colmap_util
        Eigen3::Eigen
        SQLite::SQLite3
)

COLMAP_ADD_TEST(
    NAME camera_test
    SRCS camera_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME correspondence_graph_test
    SRCS correspondence_graph_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME database_cache_test
    SRCS database_cache_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME database_test
    SRCS database_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME frame_test
    SRCS frame_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME image_test
    SRCS image_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME point2d_test
    SRCS point2d_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME point3d_test
    SRCS point3d_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME projection_test
    SRCS projection_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME reconstruction_test
    SRCS reconstruction_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME reconstruction_io_test
    SRCS reconstruction_io_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME reconstruction_manager_test
    SRCS reconstruction_manager_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME rig_test
    SRCS rig_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME scene_clustering_test
    SRCS scene_clustering_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME synthetic_test
    SRCS synthetic_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME track_test
    SRCS track_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME two_view_geometry_test
    SRCS two_view_geometry_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME visibility_pyramid_test
    SRCS visibility_pyramid_test.cc
    LINK_LIBS colmap_scene
)
