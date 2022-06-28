///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief c++ cast demo, static_cast, dynamic_cast, const_cast, reinterpret_cast
/// 
/// @file cxx_cast.cpp
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-06-27
///////////////////////////////////////////////////////////

// Current Cpp header
// System header
// C/C++ standard library header
#include <iostream>
// External library header
// Current module header
// Root directory header

void UT_Reinterpret_cast()
{
    const int size = 12;
    char c_arr[size] = {0};
    for (int i = 0; i < size; ++i)
    {
        if (i % 2 == 0)
        {
            c_arr[i] = 1;
        }
        std::cout << "index " << i << ", value = " << c_arr[i] << std::endl;
    }

    char aa = 1;
    std::cout << "aa = " << std::hex << aa << std::endl;

    auto a = reinterpret_cast<int *>(c_arr);
    auto b = reinterpret_cast<int *>(&c_arr[4]);
    auto c = reinterpret_cast<int *>(&c_arr[8]);
    
    std::cout << "a = " << *a << ", b = " << *b << ", c = " << *c << std::endl;
}

int main(int argc, char **argv)
{
    UT_Reinterpret_cast();

    return 0;
}

