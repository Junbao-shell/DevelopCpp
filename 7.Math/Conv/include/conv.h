///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief convolution header
/// 
/// @file conv.h
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-06-16
///////////////////////////////////////////////////////////

#ifndef __SOFTWARE_CONFIG_MATH_CONV_H_
#define __SOFTWARE_CONFIG_MATH_CONV_H_

// System header
// C/C++ standard library header
#include <iostream>
// External library header
#include <cuda_runtime.h>
#include <cufft.h>
// Current module header
// #include "imath.h"
// Root directory header

namespace nmath
{
enum class CONV_TYPE
{
    FULL = 0,
    SAME,
    VALID
};

class Conv // : public imath
{
public:
    Conv();
    virtual ~Conv();

    /**
     * @brief input diff size signal and kernel, retuen their convolution
     * 
     * @param signal conv(A, B): signal is the first value
     * @param kernel conv(A, B): kernel is the second value
     * @param signal_dimx the signal size in x dimension
     * @param signal_dimy the signal size in y dimension
     * @param kernel_dimx the kernel size in x dimension
     * @param kernel_dimy the kernel size in y dimension
     * @param result convolution result
     * @param type according the tye retuen diffent size, can select(FULL/SAME/VALID)
     */
    void Conv2D(const float *signal, 
                const float *kernel, 
                const int signal_dimx, 
                const int signal_dimy, 
                const int kernel_dimx, 
                const int kernel_dimy,
                float *result,
                CONV_TYPE type = CONV_TYPE::FULL);

protected:
private:
    Conv(const Conv &) = delete;
    Conv &operator=(const Conv &) = delete;

    void ForwardFFT2D(float *in, const int dimx, const int dimy, cufftComplex *out);
    void InverseFFT2D(cufftComplex *in, const int dimx, const int dimy, float *out);
    void DotFFT2D(const cufftComplex *ina, const cufftComplex *inb, const int dimx, const int dimy, cufftComplex *out);

    void PadData2D(const float *raw_signal, 
                   const int dimx, 
                   const int dimy, 
                   const int pad_dimx, 
                   const int pad_dimy, 
                   float *pad_signal);

    void GetSameData2D(const float *pad_data, 
                       const int pad_dimx, 
                       const int pad_dimy, 
                       const int dimx, 
                       const int dimy, 
                       float *same_data);

    void GetValidData2D(const float *pad_data, 
                        const int pad_dimx, 
                        const int pad_dimy, 
                        const int dimx, 
                        const int dimy, 
                        float *valid_data);

public:
protected:
private:
};
} // namespace nmath

#endif // __SOFTWARE_CONFIG_MATH_CONV_H_
