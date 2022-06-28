cmake_minimum_required(VERSION 3.10)

project(CXX_CAST)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SRC_LIST)
file(GLOB SRC_LIST ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(UT_cast ${SRC_LIST})
