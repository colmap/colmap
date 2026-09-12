# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "ui")

set(COLMAP_UI_QT_LIBS
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::OpenGL
    Qt${QT_VERSION_MAJOR}::Svg
    Qt${QT_VERSION_MAJOR}::Widgets
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    list(APPEND COLMAP_UI_QT_LIBS 
        Qt${QT_VERSION_MAJOR}::OpenGLWidgets)
endif()

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
        line_painter.h line_painter.cc
        log_widget.h log_widget.cc
        main_window.h main_window.cc
        match_matrix_widget.h match_matrix_widget.cc
        mesh_painter.h mesh_painter.cc
        model_viewer_widget.h model_viewer_widget.cc
        movie_grabber_widget.h movie_grabber_widget.cc
        options_widget.h options_widget.cc
        painter_base.h painter_base.cc
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
        ${COLMAP_UI_QT_LIBS}
)
if(MVS_ENABLED)
    target_link_libraries(colmap_ui PUBLIC colmap_mvs)
endif()
