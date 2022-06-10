// cuBLAS library demo

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#define R2C(I, J, N) ((J) * N + I)

void PrintRowMajorMatrix(const int dimx, const int dimy, const int *data)
{
    // host data
    std::cout << "row major storage" << std::endl;
    for (int i = 0; i < dimx; ++i)
    {
        std::cout << "dimx : ";
        for (int j = 0; j < dimy; ++j)
        {
            const int index = i * dimy + j;
            auto val = data[index];
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }
}

void PrintColumnMajorMatrix(const int dimx, const int dimy, const int *data)
{
    std::cout << "row major storage" << std::endl;
    for (int i = 0; i < dimx; ++i)
    {
        std::cout << "dimx : ";
        for (int j = 0; j < dimy; ++j)
        {
            const int index = j * dimx + i;
            auto val = data[index];
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }
}

void Demo1()
{
    const int dimx = 3;
    const int dimy = 4;
    const int size = dimx * dimy;
    int arr[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    PrintRowMajorMatrix(dimx, dimy, arr);
    
    // deveice data
    float *d_A;
    cudaMalloc((void**)&d_A, sizeof(float) * size);
    
    float *d_vec_A, *d_vec_B;
    cudaMalloc((void**)&d_vec_A, sizeof(float) * dimx);
    cudaMalloc((void**)&d_vec_B, sizeof(float) * dimy);

    // cuda blas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // copy 
    // cublasSetVector()

    cublasDestroy(handle);

    cudaFree(d_A);
    d_A = nullptr;
    cudaFree(d_vec_A);
    d_vec_A = nullptr;
    cudaFree(d_vec_B);
    d_vec_B = nullptr;

}

void Demo2()
{
    const int dimx = 3;
    const int dimy = 4;
    int arr[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    PrintRowMajorMatrix(dimx, dimy, arr);
    PrintColumnMajorMatrix(dimx, dimy, arr);
}

int main(int argc, char **argv)
{
    Demo2();

    return 0;
}
