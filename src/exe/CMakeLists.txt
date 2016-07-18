set(FOLDER_NAME "exe")

COLMAP_ADD_QT_EXECUTABLE(bundle_adjuster bundle_adjuster.cc)
target_link_libraries(bundle_adjuster ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(colmap colmap.cc)
target_link_libraries(colmap ${COLMAP_UI_LIBRARIES} ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(color_extractor color_extractor.cc)
target_link_libraries(color_extractor ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(database_creator database_creator.cc)
target_link_libraries(database_creator ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(exhaustive_matcher exhaustive_matcher)
target_link_libraries(exhaustive_matcher ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(feature_extractor feature_extractor.cc)
target_link_libraries(feature_extractor ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(feature_importer feature_importer.cc)
target_link_libraries(feature_importer ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(interactive_mapper interactive_mapper.cc)
target_link_libraries(interactive_mapper ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(image_registrator image_registrator.cc)
target_link_libraries(image_registrator ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(image_undistorter image_undistorter.cc)
target_link_libraries(image_undistorter ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(mapper mapper.cc)
target_link_libraries(mapper ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(matches_importer matches_importer.cc)
target_link_libraries(matches_importer ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(model_converter model_converter.cc)
target_link_libraries(model_converter ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(rig_bundle_adjuster rig_bundle_adjuster.cc)
target_link_libraries(rig_bundle_adjuster ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(sequential_matcher sequential_matcher)
target_link_libraries(sequential_matcher ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(spatial_matcher spatial_matcher.cc)
target_link_libraries(spatial_matcher ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(vocab_tree_builder vocab_tree_builder.cc)
target_link_libraries(vocab_tree_builder ${COLMAP_LIBRARIES})

COLMAP_ADD_QT_EXECUTABLE(vocab_tree_matcher vocab_tree_matcher)
target_link_libraries(vocab_tree_matcher ${COLMAP_LIBRARIES})
