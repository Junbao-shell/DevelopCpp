cmake_minimum_required(VERSION 3.10)

project(iMath VERSION 1.0)

enable_language(CUDA)

set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 1)
message("iMath Version is ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# set build type: Debug / Release
set(CMAKE_BUILD_TYPE "Debug")

# C++ compile config
set(CMAKE_CXX_FLAGS "-g -Wall -Wno-strict-aliasing -Wno-unused-variable -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

# CUDA compile config
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "found cuda module")

    set(CUDA_NVCC_FLAGS)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
    
    message(STATUS "cuda include dir: ${CUDA_INCLUDE_DIRS}")
    link_directories("/usr/local/cuda/lib64/")
    link_directories("/usr/lib/x86_64-linux-gnu/")
endif()

# Other library

# install 
set(CMAKE_INSTALL_RPATH .)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/)
link_directories(${CMAKE_SOURCE_DIR}/build)

# script
add_subdirectory(Conv)

