///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief cuda fft demo
/// 
/// @file cu_fft_demo.cu
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-06-14
///////////////////////////////////////////////////////////

// Current Cpp header
// System header
// C/C++ standard library header
#include <iostream>
#include <sstream>
// External library header
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
// Current module header
// Root directory header

typedef float2 Complex;

// static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
// static __device__ __host__ inline Complex ComplexScale(Complex, float);
// static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
// {
//     Complex c;
//     c.x = a.x * b.x - a.y + b.y;
//     c.y = a.x * b.y + a.y * b.x;
//     return c;
// }

struct ComplexMultiply
{
    ComplexMultiply(int n) : N(n) {}

    __host__ __device__ Complex operator() (const Complex &a, const Complex &b)
    {
        Complex c;
        c.x = (a.x * b.x - a.y + b.y);
        c.y = (a.x * b.y + a.y * b.x);
        return c;
    }

    int N;
};

template<typename T>
static inline std::string PrintArray(T *arr, const int size)
{
    std::stringstream stream;
    for (int i = 0; i < (size - 1); ++i)
    {
        stream << arr[i];
        stream << ' ';
    }
    stream << arr[size - 1];
    return stream.str();
}

// static inline std::string PrintArray(Complex *arr, const int size)
// {
//     std::stringstream stream;
//     for (int i = 0; i < (size - 1); ++i)
//     {
//         stream << "(" << arr[i].x << ", " << arr[i].y << ")";
//         stream << ' ';
//     }
//     stream << "(" << arr[size - 1].x << ", " << arr[size - 1].y << ")";
//     return stream.str();
// }

template<typename T>
void InitArray(const int size, T *arr)
{
    for (int i = 0; i < size; ++i)
    {
        arr[i] = i + 1;
    }
}

void InitArray(const int size, Complex *arr)
{
    for (int i = 0; i < size; ++i)
    {
        arr[i].x = i + 1;
        arr[i].y = 0;
    }
}

void InitArray(const int dimx, const int dimy, int *arr)
{
    for (int i = 0; i < dimy; ++i)
    {
        for (int j = 0; j < dimx; ++j)
        {
            const int index = i * dimx + j;
            arr[index] = i * 10 + j + 1;
        }
    }
}

void ForwardFFT(float *in, Complex *out, const int size)
{
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_R2C, 1);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);
}

void InverseFFT(Complex *in, float *out, const int size)
{
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2R, 1);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);
}

void ConvFFT(float *ina, float *inb, float *out, const int size)
{
    thrust::device_vector<Complex> d_ina(size);
    thrust::device_vector<Complex> d_inb(size);
    thrust::device_vector<Complex> d_out(size);

    Complex *raw_ina_fft = thrust::raw_pointer_cast(&d_ina[0]);
    Complex *raw_inb_fft = thrust::raw_pointer_cast(&d_inb[0]);
    Complex *raw_out_fft = thrust::raw_pointer_cast(&d_out[0]);

    std::cout << "arr a: " << ina[0] << std::endl;
    std::cout << "arr a: " << ina[1] << std::endl;
    std::cout << "arr a: " << ina[2] << std::endl;
    std::cout << "arr a: " << ina[3] << std::endl;

    ForwardFFT(ina, raw_ina_fft, size);
    ForwardFFT(inb, raw_inb_fft, size);

    thrust::transform(d_ina.begin(), d_ina.end(), d_inb.begin(), d_out.begin(), ComplexMultiply(size));

    std::cout << "arr a fft: " << d_ina[0].operator Complex().x << std::endl;
    std::cout << "arr a fft: " << d_ina[1].operator Complex().x << std::endl;
    std::cout << "arr a fft: " << d_ina[2].operator Complex().x << std::endl;
    std::cout << "arr a fft: " << d_ina[3].operator Complex().x << std::endl;

    std::cout << "arr b fft: " << d_inb[0].operator Complex().x << std::endl;
    std::cout << "arr b fft: " << d_inb[1].operator Complex().x << std::endl;
    std::cout << "arr b fft: " << d_inb[2].operator Complex().x << std::endl;
    std::cout << "arr b fft: " << d_inb[3].operator Complex().x << std::endl;

    std::cout << "arr c fft: " << d_out[0].operator Complex().x << std::endl;
    std::cout << "arr c fft: " << d_out[1].operator Complex().x << std::endl;
    std::cout << "arr c fft: " << d_out[2].operator Complex().x << std::endl;
    std::cout << "arr c fft: " << d_out[3].operator Complex().x << std::endl;

    // thrust::copy(d_out.begin(), d_out.end(), std::ostream_iterator<float2>(std::cout, " "));
    // std::cout << std::endl;

    InverseFFT(raw_out_fft, out, size);
}

void cuFFTDemo()
{
    const int size = 4;
    // const int kernel_size = 11;

    float *h_signal, *h_kernel, *h_result;
    cudaMallocHost((void**)&h_signal, sizeof(float) * size);
    cudaMallocHost((void**)&h_kernel, sizeof(float) * size);
    cudaMallocHost((void**)&h_result, sizeof(float) * size);
    
    for (int i = 0; i < size; ++i)
    {
        h_signal[i] = 1 + i;
        h_kernel[i] = 5 + i;
    }
    // InitArray<float>(size, h_signal);
    // InitArray<float>(size, h_kernel);
    memset(h_result, 0, sizeof(float) * size);
    std::cout << "signal initialize: " << PrintArray(h_signal, size) << std::endl;
    std::cout << "kernel initialize: " << PrintArray(h_kernel, size) << std::endl;

    // device memory
    float *d_signal, *d_kernel, *d_result;
    cudaMalloc((void**)&d_signal, sizeof(float) * size);
    cudaMalloc((void**)&d_kernel, sizeof(float) * size);
    cudaMalloc((void**)&d_result, sizeof(float) * size);
    
    cudaMemcpy(d_signal, h_signal, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float) * size);

    // cufft
    ConvFFT(d_signal, d_kernel, d_result, size);
    cudaMemcpy(h_result, d_result, sizeof(float) * size, cudaMemcpyDeviceToHost);

    std::cout << "conv result: " << PrintArray(h_result, size) << std::endl;

    // free memory
    cudaFree(d_signal);
    d_signal = nullptr;
    cudaFree(d_kernel);
    d_kernel = nullptr;
    cudaFree(d_result);
    d_result = nullptr;
    cudaFreeHost(h_signal);
    h_signal = nullptr;
    cudaFreeHost(h_kernel);
    h_kernel = nullptr;
    cudaFreeHost(h_result);
    h_result = nullptr;
}

int main(int argc, char **argv)
{
    cuFFTDemo();

    return 0;
}