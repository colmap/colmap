set(FOLDER_NAME "ui")

set(COLMAP_QT_MODULES Core OpenGL)

qt5_add_resources(QT_RESOURCES resources.qrc)

COLMAP_ADD_LIBRARY(ui
    bundle_adjustment_options_widget.h bundle_adjustment_options_widget.cc
    colormaps.h colormaps.cc
    database_management_widget.h database_management_widget.cc
    feature_extraction_widget.h feature_extraction_widget.cc
    feature_matching_widget.h feature_matching_widget.cc
    image_viewer_widget.h image_viewer_widget.cc
    line_painter.h line_painter.cc
    log_widget.h log_widget.cc
    main_window.h main_window.cc
    model_manager_widget.h model_manager_widget.cc
    model_stats_widget.h model_stats_widget.cc
    movie_grabber_widget.h movie_grabber_widget.cc
    new_project_widget.h new_project_widget.cc
    opengl_window.h opengl_window.cc
    options_widget.h options_widget.cc
    point_painter.h point_painter.cc
    point_viewer_widget.h point_viewer_widget.cc
    qt_utils.h qt_utils.cc
    reconstruction_options_widget.h reconstruction_options_widget.cc
    render_options_widget.h render_options_widget.cc
    triangle_painter.h triangle_painter.cc
    undistort_widget.h undistort_widget.cc
    ${QT_RESOURCES}
)
