set(FOLDER_NAME "ui")

set(COLMAP_QT_MODULES Core OpenGL)

qt5_add_resources(QT_RESOURCES resources.qrc)

COLMAP_ADD_LIBRARY(ui
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
    render_options.h render_options.cc
    render_options_widget.h render_options_widget.cc
    thread_control_widget.h thread_control_widget.cc
    triangle_painter.h triangle_painter.cc
    undistortion_widget.h undistortion_widget.cc
    ${QT_RESOURCES}
)
