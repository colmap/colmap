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


set(FOLDER_NAME "sensor")

COLMAP_ADD_LIBRARY(
    NAME colmap_sensor
    SRCS
        bitmap.h bitmap.cc
        database.h database.cc
        models.h models.cc
        rig_calib.h rig_calib.cc
        specs.h specs.cc
    PUBLIC_LINK_LIBS
        Ceres::ceres
        Eigen3::Eigen
    PRIVATE_LINK_LIBS
        colmap_geometry
        colmap_util
        colmap_vlfeat
        freeimage::FreeImage
)

COLMAP_ADD_TEST(
    NAME bitmap_test
    SRCS bitmap_test.cc
    LINK_LIBS
        colmap_sensor
        freeimage::FreeImage
)
COLMAP_ADD_TEST(
    NAME database_test
    SRCS database_test.cc
    LINK_LIBS colmap_sensor
)
COLMAP_ADD_TEST(
    NAME models_test
    SRCS models_test.cc
    LINK_LIBS colmap_sensor
)
COLMAP_ADD_TEST(
    NAME rig_calib_test
    SRCS rig_calib_test.cc
    LINK_LIBS colmap_sensor
)
