///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief convolution cuda header
/// 
/// @file conv.cuh
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-06-18
///////////////////////////////////////////////////////////

#ifndef __SOFTWARE_MATH_CONFIG_CONV_CUH_
#define __SOFTWARE_MATH_CONFIG_CONV_CUH_

// System header
// C/C++ standard library header
#include <iostream>
// External library header
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
// Current module header
// Root directory header

namespace nmath
{
template<typename T>
__global__ void cudaMatrixMul(const T *a, const T *b, const int size, T *c)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < size)
    {
        c[tidx] = a[tidx] * b[tidx];
    }
}

template<typename T>
__global__ void cudaMatrixMul(const T *a, const T *b, const int dimx, const int dimy, T *c)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidx < dimx && tidy < dimy)
    {
        const int index = tidy * dimx + tidx;
        c[index] = a[index] * b[index];
    }
}

__global__ void cudaComplexMatrixMul(const cufftComplex *a, const cufftComplex *b, const int size, cufftComplex *c)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < size)
    {
        c[tidx].x = (a[tidx].x * b[tidx].x - a[tidx].y * b[tidx].y) / size;
        c[tidx].y = (a[tidx].x * b[tidx].y + a[tidx].y * b[tidx].x) / size;
    }
}

__global__ void cudaComplexMatrixScale(cufftComplex *a, const int size, const int scale)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < size)
    {
        a[tidx].x /= scale;
        a[tidx].y /= scale;
    }
}

__global__ void cudaComplexMatrixMul(const cufftDoubleComplex *a, 
                                     const cufftDoubleComplex *b, 
                                     const int size, 
                                     cufftDoubleComplex *c)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < size)
    {
        c[tidx].x = a[tidx].x * b[tidx].x - a[tidx].y * b[tidx].y;
        c[tidx].y = a[tidx].x * b[tidx].y + a[tidx].y * b[tidx].x;
    }
}

__global__ void cudaComplexMatrixScale(cufftDoubleComplex *a, const int size, const int scale)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < size)
    {
        a[tidx].x /= scale;
        a[tidx].y /= scale;
    }
}

} // namespace nmath

#endif // __SOFTWARE_MATH_CONFIG_CONV_CUH_
