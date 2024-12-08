COLMAP_ADD_LIBRARY(
    NAME colmap_lightglue
    SRCS
        attention.cpp
        attention.hpp
        core.cpp
        core.hpp
        encoding.cpp
        encoding.hpp
        matcher.cpp
        matcher.hpp
        transformer.cpp
        transformer.hpp
        lightglue_resources.qrc
    PUBLIC_LINK_LIBS
        torch
        Qt5::Core
)
