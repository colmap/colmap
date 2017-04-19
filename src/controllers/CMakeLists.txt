set(FOLDER_NAME "controllers")

COLMAP_ADD_LIBRARY(controllers
    bundle_adjustment.h bundle_adjustment.cc
    incremental_mapper.h incremental_mapper.cc
)
