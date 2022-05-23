///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief c++ stl function demo
/// 
/// @file std_function.cpp
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-05-23
///////////////////////////////////////////////////////////

// Current Cpp header
// System header
// C/C++ standard library header
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <functional>
#include <memory>
// External library header
// Current module header
// Root directory header

namespace CaseA
{
typedef int (*func)();

int print1()
{
    printf("%s\t%d\n", __FUNCTION__, __LINE__);
    return 0;
}

int print2()
{
    printf("%s\t%d\n", __FUNCTION__, __LINE__);
    return 0;
}

int main(int argc, char **argv)
{
    func f1 = print1;
    f1();

    f1 = print2;
    f1();

    return 0;
}

} // namespace CaseA

namespace CaseB
{
int print1(int a, int b)
{
    printf("%s\t%d, num = %d, num = %d\n", __FUNCTION__, __LINE__, a, b);
    return 0;
}

int print2(int a, int b)
{
    printf("%s\t%d, num = %d, num = %d\n", __FUNCTION__, __LINE__, a, b);
    return 0;
}

int main(int argc, char **argv)
{
    int a = 1;
    int b = 1;
    std::string str = "hello";
    std::function<int(int, int)> func(&print1);
    func(a, b);

    int aa = 2;
    int bb = 2;
    std::string str_b = "function";
    func = &print2;
    func(aa, bb);

    return 0;
}
} // namespace CaseB

namespace CaseC
{
template<typename T>
void Func(T a)
{
    std::cout << "a = " << a << std::endl;
}

int main(int argc, char **argv)
{
    std::function<void(int)> func(&Func<int>);
    int a = 4;
    func(a);

    return 0;
}
} // namespace CaseC

namespace CaseD
{
auto lambda = [](int a) { std::cout << "a + 1 = " << ++a << std::endl; return a;};

int main(int argc, char **argv)
{
    std::function<int(int)> func(lambda);

    int a = 2;
    func(a);

    return 0;
}
} // namespace CaseD

int main(int argc, char **argv)
{
    CaseA::main(argc, argv);
    CaseB::main(argc, argv);
    CaseC::main(argc, argv);
    CaseD::main(argc, argv);

    return 0;
}




















