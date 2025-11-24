COLMAP_ADD_LIBRARY(
    NAME colmap_glomap
    SRCS
        controllers/global_mapper.h controllers/global_mapper.cc
        controllers/option_manager.h controllers/option_manager.cc
        controllers/rotation_averager.h controllers/rotation_averager.cc
        controllers/track_establishment.h controllers/track_establishment.cc
        controllers/track_retriangulation.h controllers/track_retriangulation.cc
        estimators/bundle_adjustment.h estimators/bundle_adjustment.cc
        estimators/global_positioning.h estimators/global_positioning.cc
        estimators/global_rotation_averaging.h estimators/global_rotation_averaging.cc
        estimators/gravity_refinement.h estimators/gravity_refinement.cc
        estimators/relpose_estimation.h estimators/relpose_estimation.cc
        estimators/rotation_initializer.h estimators/rotation_initializer.cc
        estimators/view_graph_calibration.h estimators/view_graph_calibration.cc
        io/colmap_converter.h io/colmap_converter.cc
        io/colmap_io.h io/colmap_io.cc
        io/pose_io.h io/pose_io.cc
        math/gravity.h math/gravity.cc
        math/rigid3d.h math/rigid3d.cc
        math/tree.h math/tree.cc
        math/two_view_geometry.h math/two_view_geometry.cc
        processors/image_pair_inliers.h processors/image_pair_inliers.cc
        processors/image_undistorter.h processors/image_undistorter.cc
        processors/reconstruction_normalizer.h processors/reconstruction_normalizer.cc
        processors/reconstruction_pruning.h processors/reconstruction_pruning.cc
        processors/relpose_filter.h processors/relpose_filter.cc
        processors/track_filter.h processors/track_filter.cc
        processors/view_graph_manipulation.h processors/view_graph_manipulation.cc
        scene/view_graph.h scene/view_graph.cc
        scene/frame.h
        scene/image_pair.h
        scene/image.h
        scene/track.h
        scene/types_sfm.h
        scene/types.h
    PUBLIC_LINK_LIBS
        colmap_controllers
        colmap_estimators
        colmap_exe
        colmap_feature_types
        colmap_feature
        colmap_geometry
        colmap_image
        colmap_math
        colmap_mvs
        colmap_optim
        colmap_retrieval
        colmap_scene
        colmap_sensor
        colmap_sfm
        colmap_util
        Eigen3::Eigen
        Ceres::ceres
        OpenMP::OpenMP_CXX
        PoseLib::PoseLib
        Boost::boost
)

if(MSVC)
    target_compile_options(colmap_glomap PRIVATE /bigobj)
else()
    target_compile_options(colmap_glomap PRIVATE
        -Wall
        -Wno-sign-compare
        -Wno-unused-variable
    )
endif()

COLMAP_ADD_EXECUTABLE(
    NAME glomap_main
    SRCS
        glomap.cc
        exe/global_mapper.h
        exe/global_mapper.cc
        exe/rotation_averager.h
        exe/rotation_averager.cc
    LINK_LIBS
        colmap_glomap
)
set_target_properties(glomap_main PROPERTIES OUTPUT_NAME glomap)

COLMAP_ADD_TEST(
    NAME glomap_test
    SRCS
        controllers/global_mapper_test.cc
        controllers/rotation_averager_test.cc
    LINK_LIBS
        colmap_glomap
)
