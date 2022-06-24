cmake_minimum_required(VERSION 3.10)

project(CppDesignModel)

# set build type: Debug / Release
set(CMAKE_BUILD_TYPE "Release")

# C++ compile parameters
set(CMAKE_CXX_FLAGS "-g -Wall -Wno-strict-aliasing -Wno-unused-variable -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(${CMAKE_SOURCE_DIR}/ProducerConsumer)

