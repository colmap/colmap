set(FOLDER_NAME "ext")

COLMAP_SET_THIRD_PARTY_FOLDER()

if(NOT IS_MSVC)
    string(REPLACE "-std=c++0x" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Wextra" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_subdirectory(ANNchar)
add_subdirectory(ANNfloat)
add_subdirectory(PBA)
add_subdirectory(SiftGPU)
add_subdirectory(SQLite)
add_subdirectory(VLFeat)
add_subdirectory(VocabLib)

COLMAP_UNSET_THIRD_PARTY_FOLDER()
