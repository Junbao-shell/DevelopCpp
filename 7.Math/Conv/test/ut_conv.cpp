///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief convolution excutable
/// 
/// @file ut_conv.cpp
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-06-16
///////////////////////////////////////////////////////////

// Current Cpp header
#include "conv.h"
// System header
// C/C++ standard library header
#include <iostream>
#include <sstream>
#include <string.h>
#include <memory>
#include <algorithm>
// External library header
#include <cuda_runtime.h>
// Current module header
// Root directory header
#include "istring.h"
#include "macro/cuda_check.h"

using namespace nmath;

void UT_Conv2D()
{
    const int A_dimx = 3;
    const int A_dimy = 5;
    const int B_dimx = 5;
    const int B_dimy = 8;
    const int sizeA = A_dimx * A_dimy;
    const int sizeB = B_dimx * B_dimy;

    float *h_arrA = new float[sizeA]();
    float *h_arrB = new float[sizeB]();

    float arrA[sizeA] = {80,24,70,
                         14,15,60,
                         4, 68,51,
                         46,36,78,
                         57,52,69};

    float arrB[sizeB] = {32,34,70,8, 52,
                         84,43,26,4, 39,
                         67,80,70,63,98,
                         59,15,37,94,30,
                         2, 61,55,32,60,
                         84,94,23,75,58,
                         12,64,91,29,62,
                         85,5, 38,13,57};

    memcpy(h_arrA, arrA, sizeof(float) * sizeA);
    memcpy(h_arrB, arrB, sizeof(float) * sizeB);

    float *d_arrA, *d_arrB;
    CUDACHECK(cudaMalloc((void**)&d_arrA, sizeof(float) * sizeA));
    CUDACHECK(cudaMalloc((void**)&d_arrB, sizeof(float) * sizeB));

    CUDACHECK(cudaMemcpy(d_arrA, arrA, sizeof(float) * sizeA, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_arrB, arrB, sizeof(float) * sizeB, cudaMemcpyHostToDevice));

    std::cout << "array A: " << std::endl;
    std::cout << iString::PrintArray<float>(h_arrA, A_dimx, A_dimy);

    std::cout << "array B: " << std::endl;
    std::cout << iString::PrintArray<float>(h_arrB, B_dimx, B_dimy);

    
    const int C_dimx = A_dimx + B_dimx - 1;
    const int C_dimy = A_dimy + B_dimy - 1;
    const int sizeC = C_dimx * C_dimy;
    float *res, *d_res;
    CUDACHECK(cudaMallocHost((void**)&res, sizeof(float) * sizeC));
    CUDACHECK(cudaMalloc((void**)&d_res, sizeof(float) * sizeC));

    auto conv = std::make_shared<Conv>();
    conv->Conv2D(d_arrB, d_arrA, B_dimx, B_dimy, A_dimx, A_dimy, d_res, CONV_TYPE::FULL);

    CUDACHECK(cudaMemcpy(res, d_res, sizeof(float) * sizeC, cudaMemcpyDeviceToHost));
    std::cout << "result: " << std::endl;
    std::cout << iString::PrintArray<float>(res, C_dimx, C_dimy);

    delete[] h_arrA; h_arrA = nullptr;
    delete[] h_arrB; h_arrB = nullptr;
    CUDACHECK(cudaFreeHost(res)); res = nullptr;

    CUDACHECK(cudaFree(d_arrA)); d_arrA = nullptr;
    CUDACHECK(cudaFree(d_arrB)); d_arrB = nullptr;
    CUDACHECK(cudaFree(d_res)); d_res = nullptr;
}

int main(int argc, char **argv)
{
    UT_Conv2D();

    return 0;
}

