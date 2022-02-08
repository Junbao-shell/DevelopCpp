# 

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/)

add_executable(UT_GTest main.cpp)
target_link_libraries(UT_GTest gtest glog)

