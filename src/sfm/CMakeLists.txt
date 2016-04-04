set(FOLDER_NAME "sfm")

COLMAP_ADD_LIBRARY(sfm
    controllers.h controllers.cc
    incremental_mapper.h incremental_mapper.cc
    incremental_triangulator.h incremental_triangulator.cc
)
