///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief covolution module main in CUDA cufft
/// 
/// @file conv.cu
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-06-18
///////////////////////////////////////////////////////////

// Current Cpp header
#include "conv.h"
#include "conv.cuh"
// System header
// C/C++ standard library header
#include <iostream>
#include <string.h>
// External library header
// Current module header
#include "istring.h"
// Root directory header
#include "macro/cuda_check.h"

using namespace nmath;

Conv::Conv()
{
}

Conv::~Conv()
{
}

void Conv::Conv2D(const float *signal, 
                  const float *kernel, 
                  const int signal_dimx, 
                  const int signal_dimy, 
                  const int kernel_dimx, 
                  const int kernel_dimy,
                  float *result,
                  CONV_TYPE type)
{
    const int pad_dimx = signal_dimx + kernel_dimx - 1;
    const int pad_dimy = signal_dimy + kernel_dimy - 1;
    const int new_size = pad_dimx * pad_dimy;

    float *pad_signal, *pad_kernel, *pad_result;
    CUDACHECK(cudaMalloc((void**)&pad_signal, sizeof(float) * new_size));
    CUDACHECK(cudaMemset(pad_signal, 0, sizeof(float) * new_size));
    CUDACHECK(cudaMalloc((void**)&pad_kernel, sizeof(float) * new_size));
    CUDACHECK(cudaMemset(pad_kernel, 0, sizeof(float) * new_size));
    CUDACHECK(cudaMalloc((void**)&pad_result, sizeof(float) * new_size));
    CUDACHECK(cudaMemset(pad_result, 0, sizeof(float) * new_size));

    PadData2D(signal, signal_dimx, signal_dimy, pad_dimx, pad_dimy, pad_signal);
    PadData2D(kernel, kernel_dimx, kernel_dimy, pad_dimx, pad_dimy, pad_kernel);

    CUDACHECK(cudaDeviceSynchronize());
    // create complex data
    cufftComplex *cina, *cinb, *cout;
    CUDACHECK(cudaMalloc((void**)&cina, sizeof(float) * new_size));
    CUDACHECK(cudaMemset(cina, 0, sizeof(float) * new_size));
    CUDACHECK(cudaMalloc((void**)&cinb, sizeof(float) * new_size));
    CUDACHECK(cudaMemset(cinb, 0, sizeof(float) * new_size));
    CUDACHECK(cudaMalloc((void**)&cout, sizeof(float) * new_size));
    CUDACHECK(cudaMemset(cout, 0, sizeof(float) * new_size));

    // fft forward transform
    ForwardFFT2D(pad_signal, pad_dimx, pad_dimy, cina);
    ForwardFFT2D(pad_kernel, pad_dimx, pad_dimy, cinb);

    DotFFT2D(cina, cinb, pad_dimx, pad_dimy, cout);

    InverseFFT2D(cout, pad_dimx, pad_dimy, pad_result);

    if (CONV_TYPE::SAME == type)
    {
        GetSameData2D(pad_result, pad_dimx, pad_dimy, signal_dimx, signal_dimy, result);
    }
    else if (CONV_TYPE::VALID == type)
    {
        // TODO
        // GetValidData2D(pad_result, pad_dimx, pad_dimy, signal_dimx, signal_dimy, result);
        std::cout << "get valid zoo not implement" << std::endl;
    }
    else
    {
        CUDACHECK(cudaMemcpy(result, pad_result, sizeof(float) * new_size, cudaMemcpyDeviceToDevice));
    }

    CUDACHECK(cudaFree(pad_signal)); pad_signal = nullptr;
    CUDACHECK(cudaFree(pad_kernel)); pad_kernel = nullptr;
    CUDACHECK(cudaFree(pad_result)); pad_result = nullptr;
    CUDACHECK(cudaFree(cina)); cina = nullptr;
    CUDACHECK(cudaFree(cinb)); cinb = nullptr;
    CUDACHECK(cudaFree(cout)); cout = nullptr;
}

void Conv::DotFFT2D(const cufftComplex *ina, 
                    const cufftComplex *inb, 
                    const int dimx, 
                    const int dimy, 
                    cufftComplex *out)
{
    const int size = dimx * dimy;

    dim3 Block(128);
    dim3 Grid((size + Block.x - 1) / Block.x);

    cudaComplexMatrixMul<<<Grid, Block>>>(ina, inb, size, out);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
}

void Conv::ForwardFFT2D(float *in, const int dimx, const int dimy, cufftComplex *out)
{
    cufftHandle plan;
    cufftPlan2d(&plan, dimy, dimx, CUFFT_R2C);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);
}

void Conv::InverseFFT2D(cufftComplex *in, const int dimx, const int dimy, float *out)
{
    cufftHandle plan;
    cufftPlan2d(&plan, dimy, dimx, CUFFT_C2R);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);
}

void Conv::PadData2D(const float *data, 
                     const int dimx, 
                     const int dimy, 
                     const int pad_dimx, 
                     const int pad_dimy, 
                     float *pad_data)
{
    for (int i = 0; i < dimy; ++i)
    {
        const int raw_offset = i * dimx;
        const int pad_offset = i * pad_dimx;
        CUDACHECK(cudaMemcpy(&pad_data[pad_offset], &data[raw_offset], sizeof(float) * dimx, cudaMemcpyDeviceToDevice));
    }
}

void Conv::GetSameData2D(const float *pad_data, 
                         const int pad_dimx, 
                         const int pad_dimy, 
                         const int dimx, 
                         const int dimy, 
                         float *same_data)
{
    const int offsetx = std::ceil((pad_dimx - dimx) / 2.0);
    const int offsety = std::ceil((pad_dimy - dimy) / 2.0);

    for (int i = 0; i < dimy; ++i)
    {
        const int offset_full = (i + offsety) * pad_dimx + offsetx;
        const int offset_same = i * dimx;
        CUDACHECK(cudaMemcpy(&same_data[offset_same], &pad_data[offset_full], sizeof(float) * dimx, 
            cudaMemcpyDeviceToDevice));
    }
}
