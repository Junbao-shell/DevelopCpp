# CMakeList.txt
cmake_minimum_required(VERSION 3.5)
project(test)

set(CMAKE_CXX_FLAGS -std=c++11)

include_directories(include)

set(SRC_LIST
  src/add.cpp)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/bin/)

add_library(UT_Shared STATIC ${SRC_LIST})
add_executable(test src/main.cpp)
target_link_libraries(test UT_Shared)
set_target_properties(UT_Shared PROPERTIES VERSION 1.0.1 SOVERSION 1)