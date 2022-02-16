# CMakeList.txt
cmake_minimum_required(VERSION 3.0)
project(UT_Shared)

set(CMAKE_C_COMPILER "D:/Software/MinGW/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "D:/Software/MinGW/bin/g++.exe")

set(CMAKE_CXX_FLAGS
  -std=c++11
  -g
  -G
  -Wall
  -Werror)

set(CMAKE_CUDA_FLAGS
  -g
  -G)

include_directories(include)

set(SRC_LIST
  src/add.cpp)

add_library(UT_Shared SHARED ${SRC_LIST})
set_target_properties(UT_Shared PROPERTIES VERSION 1.0.1 SOVERSION 1)