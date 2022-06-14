cmake_minimum_required(VERSION 3.10)

project(CUTHRUST)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "found cuda module")

    set(CUDA_NVCC_FLAGS)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -D_FORCE_INLINES)
    
    message(STATUS "cuda include dir: ${CUDA_INCLUDE_DIRS}")
    include_directories("/home/nv/gaojunbao/OpenSource/cuda-samples/Common/")
    link_directories("/usr/local/cuda/lib64/")
    link_directories("/usr/local/cuda/bin/")
    link_directories("/usr/lib/x86_64-linux-gnu/")
endif()

cuda_add_executable(demo cu_thrust_demo.cu)

