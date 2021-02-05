cmake_minimum_required(VERSION 2.8.12)
project(TNGBuilder)
include(ExternalProject)

set(TNG_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/tng")
set(TNG_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/tng")
set(TNG_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${TNG_INSTALL_DIR} -DBUILD_SHARED_LIBS=OFF -DCFLAGS=-fPIC -DCXXFLAGS=-fPIC)

message("TNG_PREFIX='${TNG_PREFIX}'")
message("TNG_INSTALL_DIR='${TNG_INSTALL_DIR}'")
message("TNG_CMAKE_ARGS='${TNG_CMAKE_ARGS}'")
message("GIT_EXECUTABLE='${GIT_EXECUTABLE}'")

ExternalProject_Add(tng
                    PREFIX ${TNG_PREFIX}
                    GIT_REPOSITORY https://github.com/gromacs/tng.git
		    GIT_TAG v1.7.6
                    INSTALL_DIR ${TNG_INSTALL_DIR}
                    CMAKE_ARGS ${TNG_CMAKE_ARGS} 
		    CMAKE_COMMAND CFLAGS=-fPIC CXXFLAGS=-fPIC ${CMAKE_COMMAND}
)


add_custom_target(tngio_comp ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tng
)

