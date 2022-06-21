# iMath/Conv/CMakeLists.txt

set(SUB_PROJECT_NAME Conv)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB CURR_CPP_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
cuda_add_library(${SUB_PROJECT_NAME} SHARED ${CURR_CPP_SRC_LIST} ${CURR_CUDA_SRC_LIST})
target_link_libraries(${SUB_PROJECT_NAME} cufft)

file(GLOB CURR_UT_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)
string(CONCAT APP_NAME "UT_" ${SUB_PROJECT_NAME})
cuda_add_executable(${APP_NAME} ${CURR_UT_SRC_LIST})
target_link_libraries(${APP_NAME} ${SUB_PROJECT_NAME} cufft)

# TODO
# if (CUDA_FOUND)
#     file(GLOB CURR_CUDA_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.CU)

#     cuda_add_library(${SUB_PROJECT_NAME} SHARED ${CURR_CPP_SRC_LIST} ${CURR_CUDA_SRC_LIST})
#     target_link_libraries(${SUB_PROJECT_NAME})

#     cuda_add_executable(UT_${SUB_PROJECT_NAME}, ${CURR_CPP_SRC_LIST} ${CURR_CUDA_SRC_LIST})
# else()
#     add_library(${SUB_PROJECT_NAME} SHARED ${CURR_CPP_SRC_LIST})
#     target_link_libraries(${SUB_PROJECT_NAME})

#     add_executable(UT_${SUB_PROJECT_NAME}, ${CURR_CPP_SRC_LIST}})
# endif()



