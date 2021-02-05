cmake_minimum_required(VERSION 2.8.12)
project(TNGBuilder)
include(ExternalProject)

set(TNG_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/tng")
set(TNG_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/tng")

if(Apple)
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -L /usr/local/opt/libomp/lib -I /usr/local/opt/libomp/include -fPIC")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
set(TNG_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${TNG_INSTALL_DIR} -DTNG_BUILD_TEST=OFF -DTNG_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DOpenMP_C_FLAGS=${OpenMP_C_FLAGS} -DOpenMP_C_LIB_NAMES=${OpenMP_C_LIB_NAMES} -DOpenMP_omp_LIBRARY=${OpenMP_omp_LIBRARY})
set(TNG_CMAKE_COM_ARGS CFLAGS=${OpenMP_C_FLAGS} CXXFLAGS=${OpenMP_C_FLAGS})
else()
set(TNG_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${TNG_INSTALL_DIR} -DTNG_BUILD_TEST=OFF -DTNG_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DCFLAGS=-fPIC -DCXXFLAGS=-fPIC)
set(TNG_CMAKE_COM_ARGS CFLAGS=-fPIC CXXFLAGS=-fPIC)
endif()


message("TNG_PREFIX='${TNG_PREFIX}'")
message("TNG_INSTALL_DIR='${TNG_INSTALL_DIR}'")
message("TNG_CMAKE_ARGS='${TNG_CMAKE_ARGS}'")
message("GIT_EXECUTABLE='${GIT_EXECUTABLE}'")

ExternalProject_Add(tng
                    PREFIX ${TNG_PREFIX}
                    GIT_REPOSITORY https://github.com/gromacs/tng.git
		    #GIT_TAG v1.7.6
                    INSTALL_DIR ${TNG_INSTALL_DIR}
                    CMAKE_ARGS ${TNG_CMAKE_ARGS} 
		    CMAKE_COMMAND ${TNG_CMAKE_COM_ARGS} ${CMAKE_COMMAND}
)


add_custom_target(tngio_comp ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tng
)

if (EXISTS ${TNG_INSTALL_DIR}/lib64)
ADD_CUSTOM_COMMAND(TARGET tng
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${TNG_INSTALL_DIR}/lib64 ${TNG_INSTALL_DIR}/lib 
                DEPENDS ${TNG_INSTALL_DIR}/lib64
                COMMENT "Copying tng/lib64 to tng/lib"
                )
endif()

