set(FOLDER_NAME "ext")

COLMAP_SET_THIRD_PARTY_FOLDER()

if(NOT IS_MSVC)
    string(REPLACE "-std=c++11" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_subdirectory(FLANN)
add_subdirectory(Graclus)
add_subdirectory(LSD)
add_subdirectory(PBA)
add_subdirectory(PoissonRecon)
add_subdirectory(SiftGPU)
add_subdirectory(SQLite)
add_subdirectory(VLFeat)

COLMAP_UNSET_THIRD_PARTY_FOLDER()
