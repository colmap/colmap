set(FOLDER_NAME "exe")

COLMAP_ADD_EXECUTABLE(colmap colmap.cc)

if(CUDA_ENABLED)
    target_include_directories(colmap PUBLIC ${CUDA_INCLUDE_DIRS})
endif()
