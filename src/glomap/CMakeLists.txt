if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
endif()

if(COVERAGE_ENABLED)
    add_compile_options(-coverage -fprofile-update=atomic)
    add_link_options(-coverage)
endif()

add_subdirectory(scene)
add_subdirectory(math)
add_subdirectory(io)
add_subdirectory(estimators)
add_subdirectory(processors)
add_subdirectory(controllers)
add_subdirectory(exe)

COLMAP_ADD_EXECUTABLE(
    NAME glomap_main
    SRCS
        glomap.cc
    LINK_LIBS
        colmap_glomap_exe
)
set_target_properties(glomap_main PROPERTIES OUTPUT_NAME glomap)
