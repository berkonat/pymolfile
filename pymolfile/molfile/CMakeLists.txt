cmake_minimum_required(VERSION 2.8.12)

project(molfile)

add_subdirectory(external)
add_subdirectory(molfile_plugins/compile)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_BINARY_DIR}/molfile_plugins/compile/")

add_dependencies( molfile_plugins_comp tngio_comp )

