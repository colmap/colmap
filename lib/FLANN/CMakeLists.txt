if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

COLMAP_ADD_STATIC_LIBRARY(flann
    flann.hpp flann.cpp
    ext/lz4.h ext/lz4.c
    ext/lz4hc.h ext/lz4hc.c
)
