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

set(FOLDER_NAME "base")

COLMAP_ADD_SOURCES(
    alignment.h alignment.cc
    camera.h camera.cc
    camera_rig.h camera_rig.cc
    correspondence_graph.h correspondence_graph.cc
    database.h database.cc
    database_cache.h database_cache.cc
    image.h image.cc
    point2d.h
    point3d.h point3d.cc
    reconstruction.h reconstruction.cc
    reconstruction_manager.h reconstruction_manager.cc
    scene_clustering.h scene_clustering.cc
    synthetic.h synthetic.cc
    track.h track.cc
    visibility_pyramid.h visibility_pyramid.cc
)

COLMAP_ADD_TEST(camera_rig_test camera_rig_test.cc)
COLMAP_ADD_TEST(camera_test camera_test.cc)
COLMAP_ADD_TEST(correspondence_graph_test correspondence_graph_test.cc)
COLMAP_ADD_TEST(cost_functions_test cost_functions_test.cc)
COLMAP_ADD_TEST(database_cache_test database_cache_test.cc)
COLMAP_ADD_TEST(database_test database_test.cc)
COLMAP_ADD_TEST(image_test image_test.cc)
COLMAP_ADD_TEST(point2d_test point2d_test.cc)
COLMAP_ADD_TEST(point3d_test point3d_test.cc)
COLMAP_ADD_TEST(reconstruction_test reconstruction_test.cc)
COLMAP_ADD_TEST(reconstruction_manager_test reconstruction_manager_test.cc)
COLMAP_ADD_TEST(scene_clustering_test scene_clustering_test.cc)
COLMAP_ADD_TEST(synthetic_test synthetic_test.cc)
COLMAP_ADD_TEST(track_test track_test.cc)
COLMAP_ADD_TEST(visibility_pyramid_test visibility_pyramid_test.cc)
