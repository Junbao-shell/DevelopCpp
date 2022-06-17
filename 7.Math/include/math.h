///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief Math library interface header
/// 
/// @file math.h
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-06-16
///////////////////////////////////////////////////////////

#ifndef __SOFTWARE-CONFIG_MATH_H_
#define __SOFTWARE-CONFIG_MATH_H_

// System header
// C/C++ standard library header
#include <iostream>
// External library header
// Current module header
// Root directory header

class imath
{
public:
    imath();
    ~imath();

    enum class CONV_TYPE
    {
        FULL = 0,
        SAME,
        VALID
    };

    virtual void Conv2D(const float *signal, 
                        const float *kernel, 
                        const int signal_dimx, 
                        const int signal_dimy, 
                        const int kernel_dimx, 
                        const int kernel_dimy,
                        float *result,
                        CONV_TYPE type = CONV_TYPE::FULL) = 0;

protected:
private:
    imath(const imath &) = delete;
    imath &operator=(const imath &) = delete;

public:
protected:
private:
};

#endif // __SOFTWARE-CONFIG_MATH_H_
