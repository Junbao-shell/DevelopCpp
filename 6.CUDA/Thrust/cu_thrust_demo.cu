///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief CUDA Thrust demo
/// 
/// @file cu_thrust_demo.cu
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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
// #include <thrust/replace.h>
// Current module header
// Root directory header

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

void UT_Vector()
{
    const int size = 4;
    thrust::device_vector<float> d_vec(size);

    for (int i = 0; i < size; ++i)
    {
        d_vec[i] = i + 1;
    }

    std::cout << "init the device vector" << std::endl;
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "push new element the device vector" << std::endl;
    for (int i = 0; i < size; ++i)
    {
        d_vec.push_back(i + 11);
    }
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
}

void UT_Random()
{
    const int size = 10;
    thrust::device_vector<int> d_vec(size);
    thrust::default_random_engine gen;
    thrust::generate(d_vec.begin(), d_vec.end(), gen);

    std::cout << "random number: " << std::endl;
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    thrust::sort(d_vec.begin(), d_vec.end());
    std::cout << "sort numbers: " << std::endl;
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

}

int main(int argc, char **argv)
{
    std::cout << "========================================================" << std::endl;

    // UT_Vector();

    UT_Random();

    std::cout << "========================================================" << std::endl;

    const int size = 4;
    thrust::host_vector<int> h_vec(size);

    for (int i = 0; i < size; ++i)
    {
        h_vec[i] = i + 1;
    }

    std::cout << "thrust host vector: " << PrintArray<int>(h_vec.data(), size) << std::endl;

    thrust::device_vector<int> d_vec = h_vec;
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // thrust::fill algorithm
    thrust::device_vector<int> d_mod(size);
    thrust::fill(d_mod.begin(), d_mod.end(), 2);
    std::cout << "d_mod: ";
    thrust::copy(d_mod.begin(), d_mod.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // thrust::replace algorithm
    thrust::replace(d_vec.begin(), d_vec.end(), 3, 33);
    std::cout << "d_vec replace : ";
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // thrust::transform algorithm
    thrust::device_vector<int> d_res(size);
    thrust::transform(d_vec.begin(), d_vec.end(), d_mod.begin(), d_res.begin(), thrust::modulus<int>());
    std::cout << "thrust::transform ";
    thrust::copy(d_res.begin(), d_res.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // thrust::reduce
    thrust::device_vector<int> d_plus(size);
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
    std::cout << "d_vec sum: " << sum << std::endl;

    return 0;
}



