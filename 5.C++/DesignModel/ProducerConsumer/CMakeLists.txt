# 5.c++/DesignModel/ProducerConsumer/CMakeLists.txt

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB CURR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# add_library(ProducerConsumer SHARED ${CURR_SRC})

file(GLOB EXE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)

add_executable(UT_ProducerConsumer ${EXE_SRC} )
target_link_libraries(UT_ProducerConsumer pthread)

