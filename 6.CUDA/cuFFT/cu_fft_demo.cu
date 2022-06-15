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
#include <utility>
#include <chrono>
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

struct ComplexMultiply
{
    ComplexMultiply(int n) : N(n) {}

    __host__ __device__ cufftComplex operator() (const cufftComplex &a, const cufftComplex &b)
    {
        cufftComplex c;
        c.x = (a.x * b.x - a.y * b.y) / N;
        c.y = (a.x * b.y + a.y * b.x) / N;
        return c;
    }

    int N;
};

static __global__ void ComplexMulti(const cufftComplex *a, const cufftComplex *b, cufftComplex *c, const int size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        c[tid].x = (a[tid].x * b[tid].x - a[tid].y * b[tid].y) / size;
        c[tid].y = (a[tid].x * b[tid].y + a[tid].y * b[tid].x) / size;
    }
}

static __global__ void ComplexMulti2D(const cufftComplex *a, const cufftComplex *b, cufftComplex *c, const int dimx, const int dimy)
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

template<typename T>
static inline std::string PrintGpuArray(T *arr, const int size)
{
    T *h_arr;
    cudaMallocHost((void**)&h_arr, sizeof(T) * size);
    cudaMemcpy(h_arr, arr, sizeof(T) * size, cudaMemcpyDeviceToHost);

    std::stringstream stream;
    for (int i = 0; i < (size - 1); ++i)
    {
        stream << h_arr[i];
        stream << ' ';
    }
    stream << h_arr[size - 1];

    cudaFreeHost(h_arr);
    h_arr = nullptr;

    return stream.str();
}

void ForwardFFT1D(float *in, cufftComplex *out, const int size)
{
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_R2C, 1);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);
}

void InverseFFT1D(cufftComplex *in, float *out, const int size)
{
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2R, 1);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);
}

void ForwardFFT2D(float *in, cufftComplex *out, const int dimx, const int dimy)
{
    cufftHandle plan;
    cufftPlan2d(&plan, dimy, dimx, CUFFT_R2C);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);
}

void InverseFFT2D(cufftComplex *in, float *out, const int dimx, const int dimy)
{
    cufftHandle plan;
    cufftPlan2d(&plan, dimy, dimx, CUFFT_C2R);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);
}

void Conv2DFFT(float *ina, float *inb, float *out, const int dimx, const int dimy)
{
    const int size = dimx * dimy;
    cufftComplex *c_ina, *c_inb, *c_out;
    cudaMalloc((void**)&c_ina, sizeof(cufftComplex) * size);
    cudaMalloc((void**)&c_inb, sizeof(cufftComplex) * size);
    cudaMalloc((void**)&c_out, sizeof(cufftComplex) * size);

    std::cout << PrintGpuArray<float>(ina, size) << std::endl;

    ForwardFFT2D(ina, c_ina, dimx, dimy);
    ForwardFFT2D(inb, c_inb, dimx, dimy);
    cudaDeviceSynchronize();

    cudaMemset(ina, 0, sizeof(float) * size);
    InverseFFT2D(c_ina, ina, dimx, dimy);

    std::cout << PrintGpuArray<float>(ina, size) << std::endl;

    cufftComplex *h_ina, *h_inb, *h_out;
    cudaMallocHost((void**)&h_ina, sizeof(cufftComplex) * size);
    cudaMallocHost((void**)&h_inb, sizeof(cufftComplex) * size);
    cudaMallocHost((void**)&h_out, sizeof(cufftComplex) * size);

    cudaMemcpy(h_ina, c_ina, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inb, c_inb, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, c_out, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);

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

void PadData1D(const int raw_size, const int new_size, const float *raw_signal, float *new_signal)
{
    memcpy(new_signal, raw_signal, sizeof(float) * raw_size);
}

void PadData2D(const int dimx, const int dimy, const float *raw_signal, float *new_signal)
{
    const int new_dimx = 2 * dimx - 1;
    for (int i = 0; i < dimy; ++i)
    {
        const int raw_offset = i * dimx;
        const int new_offset = i * new_dimx;
        memcpy(&new_signal[new_offset], &raw_signal[raw_offset], sizeof(float) * dimx);
    }
}

void GetSameData(const int dimx, const int dimy, const float *pad_data, float *same_data)
{
    const int new_dimx = 2 * dimx - 1;
    const int offsetx = std::ceil((dimx - 1) / 2.0);
    const int offsety = std::ceil((dimy - 1) / 2.0);

    for (int i = 0; i < dimy; ++i)
    {
        const int offset_full = (i + offsety) * new_dimx + offsetx;
        const int offset_same = i * dimx;
        memcpy(&same_data[offset_same], &pad_data[offset_full], sizeof(float) * dimx);
    }
}

void Conv1DFFT(float *ina, float *inb, float *out, const int size)
{
    cufftComplex *c_ina, *c_inb, *c_out;
    cudaMalloc((void**)&c_ina, sizeof(cufftComplex) * size);
    cudaMalloc((void**)&c_inb, sizeof(cufftComplex) * size);
    cudaMalloc((void**)&c_out, sizeof(cufftComplex) * size);

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

void Conv2D(float *signal, float *kernel, const int dimx, const int dimy, float *result)
{
    const int new_dimx = 2 * dimx - 1;
    const int new_dimy = 2 * dimy - 1;
    const int new_size = new_dimx * new_dimy;

    float *h_new_signal, *h_new_kernel;
    cudaMallocHost((void**)&h_new_signal, sizeof(float) * new_size); memset(h_new_signal, 0, sizeof(float) * new_size);
    cudaMallocHost((void**)&h_new_kernel, sizeof(float) * new_size); memset(h_new_kernel, 0, sizeof(float) * new_size);

    PadData2D(dimx, dimy, signal, h_new_signal);
    PadData2D(dimx, dimy, kernel, h_new_kernel);

    float *d_signal, *d_kernel, *d_result;
    cudaMalloc((void**)&d_signal, sizeof(float) * new_size); cudaMemset(d_signal, 0, sizeof(float) * new_size);
    cudaMalloc((void**)&d_kernel, sizeof(float) * new_size); cudaMemset(d_kernel, 0, sizeof(float) * new_size);
    cudaMalloc((void**)&d_result, sizeof(float) * new_size); cudaMemset(d_result, 0, sizeof(float) * new_size);
    
    cudaMemcpy(d_signal, h_new_signal, sizeof(float) * new_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_new_kernel, sizeof(float) * new_size, cudaMemcpyHostToDevice);

    Conv2DFFT(d_signal, d_kernel, d_result, new_dimx, new_dimy);

    float *h_result;
    cudaMallocHost((void**)&h_result, sizeof(float) * new_size); 
    memset(h_result, 0, sizeof(float) * new_size);
    cudaMemcpy(h_result, d_result, sizeof(float) * new_size, cudaMemcpyDeviceToHost);

    GetSameData(dimx, dimy, h_result, result);

    cudaFree(d_signal);
    d_signal = nullptr;
    cudaFree(d_kernel);
    d_kernel = nullptr;
    cudaFree(d_result);
    d_result = nullptr;

    cudaFreeHost(h_new_signal);
    h_new_signal = nullptr;
    cudaFreeHost(h_new_kernel);
    h_new_kernel = nullptr;
    cudaFreeHost(h_result);
    h_result = nullptr;
}

void cuConvDemo2D()
{
    const int dimx = 4;
    const int dimy = 2;
    const int size = dimx * dimy;

    float *signal, *kernel, *result;
    cudaMallocHost((void**)&signal, sizeof(float) * size); memset(signal, 0, sizeof(float) * size);
    cudaMallocHost((void**)&kernel, sizeof(float) * size); memset(kernel, 0, sizeof(float) * size);
    cudaMallocHost((void**)&result, sizeof(float) * size); memset(result, 0, sizeof(float) * size);

    for (int i = 0; i < dimx; ++i)
    {
        for (int j = 0; j < dimy; ++j)
        {
            const int index = j * dimx + i;
            signal[index] = i + 1;
            kernel[index] = i + 1;
        }
    }

    for (int i = 0; i < dimy; ++i)
    {
        const int offset = i * dimx;
        std::cout << "conv result: " << PrintArray(signal + offset, dimx) << std::endl;
    }

    // FILE *fp;
    // fp = fopen("./down_image.raw", "rb");
    // fread(signal, sizeof(float), size, fp);
    // fclose(fp);
    
    // fp = fopen("./Func0.raw", "rb");
    // fread(kernel, sizeof(float), size, fp);
    // fclose(fp);

    auto start = std::chrono::steady_clock::now();

    Conv2D(signal, kernel, dimx, dimy, result);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "conv fft time: " << time << " ms" << std::endl;

    for (int i = 0; i < dimy; ++i)
    {
        const int offset = i * dimx;
        std::cout << "conv result: " << PrintArray(result + offset, dimx) << std::endl;
    }

    // free memory
    cudaFreeHost(signal);
    signal = nullptr;
    cudaFreeHost(kernel);
    kernel = nullptr;
    cudaFreeHost(result);
    result = nullptr;
}

void cuFFTDemo1D()
{
    const int size = 4;
    const int new_size = 2 * size - 1;
    // const int kernel_size = 11;

    float *h_signal, *h_kernal, *h_result;
    cudaMallocHost((void**)&h_signal, sizeof(float) * new_size); memset(h_signal, 0, sizeof(float) * new_size);
    cudaMallocHost((void**)&h_kernal, sizeof(float) * new_size); memset(h_kernal, 0, sizeof(float) * new_size);
    cudaMallocHost((void**)&h_result, sizeof(float) * new_size); memset(h_result, 0, sizeof(float) * new_size);
    
    for (int i = 0; i < size; ++i)
    {
        h_signal[i] = 1 + i;
        h_kernal[i] = 5 + i;
    }

    std::cout << "signal initialize: " << PrintArray(h_signal, size) << std::endl;
    std::cout << "kernel initialize: " << PrintArray(h_kernal, size) << std::endl;

    // device memory
    float *d_signal, *d_kernel, *d_result;
    cudaMalloc((void**)&d_signal, sizeof(float) * new_size); 
    cudaMemset(d_signal, 0, sizeof(float) * new_size);
    cudaMalloc((void**)&d_kernel, sizeof(float) * new_size);
    cudaMemset(d_kernel, 0, sizeof(float) * new_size);
    cudaMalloc((void**)&d_result, sizeof(float) * new_size);
    cudaMemset(d_result, 0, sizeof(float) * new_size);
    
    cudaMemcpy(d_signal, h_signal, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernal, sizeof(float) * size, cudaMemcpyHostToDevice);

    // cufft
    Conv1DFFT(d_signal, d_kernel, d_result, new_size);
    cudaMemcpy(h_result, d_result, sizeof(float) * new_size, cudaMemcpyDeviceToHost);

    std::cout << "conv result: " << PrintArray(h_result, new_size) << std::endl;

    // free memory
    cudaFree(d_signal);
    d_signal = nullptr;
    cudaFree(d_kernel);
    d_kernel = nullptr;
    cudaFree(d_result);
    d_result = nullptr;
    cudaFreeHost(h_signal);
    h_signal = nullptr;
    cudaFreeHost(h_kernal);
    h_kernal = nullptr;
    cudaFreeHost(h_result);
    h_result = nullptr;
}

int main(int argc, char **argv)
{
    // cuFFTDemo1D();

    cuConvDemo2D();

    return 0;
}