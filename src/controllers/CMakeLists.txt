set(FOLDER_NAME "controllers")

COLMAP_ADD_LIBRARY(controllers
    automatic_reconstruction.h automatic_reconstruction.cc
    bundle_adjustment.h bundle_adjustment.cc
    hierarchical_mapper.h hierarchical_mapper.cc
    incremental_mapper.h incremental_mapper.cc
)
