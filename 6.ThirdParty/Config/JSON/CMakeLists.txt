# 6.ThirdParty/Config/Json/CMakeLists.txt
cmake_minimum_required(VERSION 3.5)

project(config_json)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(Json_demo demo.cpp)