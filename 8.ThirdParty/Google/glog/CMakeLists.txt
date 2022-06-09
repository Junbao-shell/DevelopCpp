# 6.ThirdParty/CMakeLists.txt
cmake_minimum_required(VERSION 3.5)

project(GLOG_Test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
link_directories(D:/Software/LLVM/llvm-mingw-20211002-msvcrt-x86_64/bin)

add_executable(GLOG_Test main.cpp)
target_link_libraries(GLOG_Test glog unwind)
