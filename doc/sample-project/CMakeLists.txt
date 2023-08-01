cmake_minimum_required(VERSION 3.10)

project(SampleProject)

find_package(colmap REQUIRED)
# or to require a specific version: find_package(colmap 3.4 REQUIRED)

add_executable(hello_world hello_world.cc)
target_link_libraries(hello_world colmap::colmap)
