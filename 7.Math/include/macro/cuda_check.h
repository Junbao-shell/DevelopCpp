///////////////////////////////////////////////////////////
/// @copyright copyright description
///
/// @brief cuda check macro 
///
/// @file cuda_check.h
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
///
/// @date 2022-06-18
///////////////////////////////////////////////////////////

#ifndef __INCLUDE_CUDA_CHECK_H_
#define __INCLUDE_CUDA_CHECK_H_

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDACHECK(call)                                            \
  do                                                               \
  {                                                                \
    const cudaError_t error_code = call;                           \
    if (error_code != cudaSuccess)                                 \
    {                                                              \
      printf("CUDA Error:\n");                                     \
      printf(" File: %s\n", __FILE__);                             \
      printf(" Line: %d\n", __LINE__);                             \
      printf(" Error code: %d\n", error_code);                     \
      printf(" Error text: %s\n", cudaGetErrorString(error_code)); \
    }                                                              \
  } while (0)

#ifdef STRONG_DEBUG
#define CUDA_CHECK_KERNEL               \
  {                                     \
    CUDACHECK(cudaGetLastError());      \
    CUDACHECK(cudaDeviceSynchronize()); \
  }
#else
#define CUDA_CHECK_KERNEL          \
  {                                \
    CUDACHECK(cudaGetLastError()); \
  }
#endif // STRONG_DEBUG

#endif // __INCLUDE_CUDA_CHECK_H_