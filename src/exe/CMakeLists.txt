set(FOLDER_NAME "exe")

COLMAP_ADD_EXECUTABLE(automatic_reconstructor automatic_reconstructor.cc)

COLMAP_ADD_EXECUTABLE(bundle_adjuster bundle_adjuster.cc)

COLMAP_ADD_EXECUTABLE(colmap colmap.cc)

COLMAP_ADD_EXECUTABLE(color_extractor color_extractor.cc)

COLMAP_ADD_EXECUTABLE(database_creator database_creator.cc)

COLMAP_ADD_EXECUTABLE(dense_fuser dense_fuser.cc)

COLMAP_ADD_EXECUTABLE(dense_mesher dense_mesher.cc)

COLMAP_ADD_EXECUTABLE(exhaustive_matcher exhaustive_matcher.cc)

COLMAP_ADD_EXECUTABLE(feature_extractor feature_extractor.cc)

COLMAP_ADD_EXECUTABLE(feature_importer feature_importer.cc)

COLMAP_ADD_EXECUTABLE(image_rectifier image_rectifier.cc)

COLMAP_ADD_EXECUTABLE(image_registrator image_registrator.cc)

COLMAP_ADD_EXECUTABLE(image_undistorter image_undistorter.cc)

COLMAP_ADD_EXECUTABLE(mapper mapper.cc)

COLMAP_ADD_EXECUTABLE(matches_importer matches_importer.cc)

COLMAP_ADD_EXECUTABLE(model_aligner model_aligner.cc)

COLMAP_ADD_EXECUTABLE(model_analyzer model_analyzer.cc)

COLMAP_ADD_EXECUTABLE(model_converter model_converter.cc)

COLMAP_ADD_EXECUTABLE(model_merger model_merger.cc)

COLMAP_ADD_EXECUTABLE(model_orientation_aligner model_orientation_aligner.cc)

COLMAP_ADD_EXECUTABLE(point_triangulator point_triangulator.cc)

COLMAP_ADD_EXECUTABLE(rig_bundle_adjuster rig_bundle_adjuster.cc)

COLMAP_ADD_EXECUTABLE(sequential_matcher sequential_matcher)

COLMAP_ADD_EXECUTABLE(spatial_matcher spatial_matcher.cc)

COLMAP_ADD_EXECUTABLE(transitive_matcher transitive_matcher.cc)

COLMAP_ADD_EXECUTABLE(vocab_tree_builder vocab_tree_builder.cc)

COLMAP_ADD_EXECUTABLE(vocab_tree_matcher vocab_tree_matcher.cc)

COLMAP_ADD_EXECUTABLE(vocab_tree_retriever vocab_tree_retriever.cc)

if(CUDA_ENABLED)
    COLMAP_ADD_EXECUTABLE(dense_stereo dense_stereo.cc)
    target_include_directories(dense_stereo PUBLIC ${CUDA_INCLUDE_DIRS})
endif()
