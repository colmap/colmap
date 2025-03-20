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


set(FOLDER_NAME "ui")

COLMAP_ADD_LIBRARY(
    NAME colmap_ui
    SRCS
        automatic_reconstruction_widget.h automatic_reconstruction_widget.cc
        bundle_adjustment_widget.h bundle_adjustment_widget.cc
        colormaps.h colormaps.cc
        database_management_widget.h database_management_widget.cc
        dense_reconstruction_widget.h dense_reconstruction_widget.cc
        feature_extraction_widget.h feature_extraction_widget.cc
        feature_matching_widget.h feature_matching_widget.cc
        image_viewer_widget.h image_viewer_widget.cc
        license_widget.h license_widget.cc
        line_painter.h line_painter.cc
        log_widget.h log_widget.cc
        main_window.h main_window.cc
        match_matrix_widget.h match_matrix_widget.cc
        model_viewer_widget.h model_viewer_widget.cc
        movie_grabber_widget.h movie_grabber_widget.cc
        options_widget.h options_widget.cc
        point_painter.h point_painter.cc
        point_viewer_widget.h point_viewer_widget.cc
        project_widget.h project_widget.cc
        qt_utils.h qt_utils.cc
        reconstruction_manager_widget.h reconstruction_manager_widget.cc
        reconstruction_options_widget.h reconstruction_options_widget.cc
        reconstruction_stats_widget.h reconstruction_stats_widget.cc
        render_options.h
        render_options_widget.h render_options_widget.cc
        thread_control_widget.h thread_control_widget.cc
        triangle_painter.h triangle_painter.cc
        undistortion_widget.h undistortion_widget.cc
        resources.qrc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_image
        colmap_scene
        colmap_controllers
        Qt5::Core
        Qt5::OpenGL
        Qt5::Widgets
)
