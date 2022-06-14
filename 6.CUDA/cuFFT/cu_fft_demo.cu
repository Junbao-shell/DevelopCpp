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
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
// Current module header
// Root directory header

typedef float2 Complex;

struct ComplexMultiply
{
    ComplexMultiply(int n) : N(n) {}

    __host__ __device__ Complex operator() (const Complex &a, const Complex &b)
    {
        Complex c;
        c.x = (a.x * b.x - a.y * b.y) / N;
        c.y = (a.x * b.y + a.y * b.x) / N;
        return c;
    }

    int N;
};

static __global__ void ComplexMulti(const Complex *a, const Complex *b, Complex *c, const int size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        c[tid].x = (a[tid].x * b[tid].x - a[tid].y * b[tid].y) / size;
        c[tid].y = (a[tid].x * b[tid].y + a[tid].y * b[tid].x) / size;
    }
}

static __global__ void ComplexMulti2D(const Complex *a, const Complex *b, Complex *c, const int dimx, const int dimy)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidx < dimx && tidy < dimy)
    {
        const int i = tidy * dimx + tidx;

        c[i].x = (a[i].x * b[i].x - a[i].y * b[i].y) / (dimx * dimy);
        c[i].y = (a[i].x * b[i].y + a[i].y * b[i].x) / (dimx * dimy);
    }
}

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

void ForwardFFT1D(float *in, Complex *out, const int size)
{
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_R2C, 1);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);
}

void InverseFFT1D(Complex *in, float *out, const int size)
{
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2R, 1);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);
}

void ForwardFFT2D(float *in, Complex *out, const int dimx, const int dimy)
{
    cufftHandle plan;
    cufftPlan2d(&plan, dimy, dimx, CUFFT_R2C);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);
}

void InverseFFT2D(Complex *in, float *out, const int dimx, const int dimy)
{
    cufftHandle plan;
    cufftPlan2d(&plan, dimy, dimx, CUFFT_C2R);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);
}

void Conv2DFFT(float *ina, float *inb, float *out, const int dimx, const int dimy)
{
    const int size = dimx * dimy;
    Complex *c_ina, *c_inb, *c_out;
    cudaMalloc((void**)&c_ina, sizeof(Complex) * size);
    cudaMalloc((void**)&c_inb, sizeof(Complex) * size);
    cudaMalloc((void**)&c_out, sizeof(Complex) * size);

    ForwardFFT2D(ina, c_ina, dimx, dimy);
    ForwardFFT2D(inb, c_inb, dimx, dimy);
    cudaDeviceSynchronize();


    Complex *h_ina, *h_inb, *h_out;
    cudaMallocHost((void**)&h_ina, sizeof(Complex) * size);
    cudaMallocHost((void**)&h_inb, sizeof(Complex) * size);
    cudaMallocHost((void**)&h_out, sizeof(Complex) * size);

    cudaMemcpy(h_ina, c_ina, sizeof(Complex) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inb, c_inb, sizeof(Complex) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, c_out, sizeof(Complex) * size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; ++i)
    {
        std::cout << "index: " << i << " (" << h_ina[i].x << ", " << h_ina[i].y << ")" << std::endl;
    }

    dim3 Block(32, 16);
    dim3 Grid((dimx + Block.x - 1) / Block.x, (dimy + Block.y - 1) / Block.y);
    ComplexMulti2D<<<Grid, Block>>>(c_ina, c_inb, c_out, dimx, dimy);
    cudaDeviceSynchronize();

    InverseFFT2D(c_out, out, dimx, dimy);
    cudaDeviceSynchronize();

    cudaFree(c_ina);
    c_ina = nullptr;
    cudaFree(c_inb);
    c_inb = nullptr;
    cudaFree(c_out);
    c_out = nullptr;
    
    cudaFreeHost(h_ina);
    h_ina = nullptr;
    cudaFreeHost(h_inb);
    h_inb = nullptr;
    cudaFreeHost(h_out);
    h_out = nullptr;
}

void Conv1DFFT(float *ina, float *inb, float *out, const int size)
{
    Complex *c_ina, *c_inb, *c_out;
    cudaMalloc((void**)&c_ina, sizeof(Complex) * size);
    cudaMalloc((void**)&c_inb, sizeof(Complex) * size);
    cudaMalloc((void**)&c_out, sizeof(Complex) * size);

    ForwardFFT1D(ina, c_ina, size);
    ForwardFFT1D(inb, c_inb, size);
    cudaDeviceSynchronize();

    dim3 Block(128);
    dim3 Grid((size + Block.x - 1) / Block.x);
    ComplexMulti<<<Grid, Block>>>(c_ina, c_inb, c_out, size);
    cudaDeviceSynchronize();

    InverseFFT1D(c_out, out, size);
    cudaDeviceSynchronize();

    cudaFree(c_ina);
    c_ina = nullptr;
    cudaFree(c_inb);
    c_inb = nullptr;
    cudaFree(c_out);
    c_out = nullptr;
}

void cuFFTDemo2D()
{
    const int dimx = 4;
    const int dimy = 2;
    const int size = dimx * dimy;

    float *h_signal, *h_kernel, *h_result;
    cudaMallocHost((void**)&h_signal, sizeof(float) * size);
    cudaMallocHost((void**)&h_kernel, sizeof(float) * size);
    cudaMallocHost((void**)&h_result, sizeof(float) * size);
    
    for (int i = 0; i < size; ++i)
    {
        h_signal[i] = 1 + i;
        h_kernel[i] = 1 + i;
    }

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
    Conv2DFFT(d_signal, d_kernel, d_result, dimx, dimy);
    cudaMemcpy(h_result, d_result, sizeof(float) * size, cudaMemcpyDeviceToHost);

    std::cout << "conv result: " << PrintArray(h_result, dimx) << std::endl;
    std::cout << "conv result: " << PrintArray(h_result + dimx, dimx) << std::endl;

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

void cuFFTDemo1D()
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
    Conv1DFFT(d_signal, d_kernel, d_result, size);
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
    // cuFFTDemo1D();

    cuFFTDemo2D();

    return 0;
}