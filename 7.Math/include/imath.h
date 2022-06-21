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

#ifndef __SOFTWARE_CONFIG_MATH_H_
#define __SOFTWARE_CONFIG_MATH_H_

// System header
// C/C++ standard library header
#include <iostream>
// External library header
// Current module header
// Root directory header

namespace nmath
{

class imath
{
public:
    imath() = default;
    virtual ~imath() {}

    virtual void Conv2D(const float *signal, 
                        const float *kernel, 
                        const int signal_dimx, 
                        const int signal_dimy, 
                        const int kernel_dimx, 
                        const int kernel_dimy,
                        float *result,
                        CONV_TYPE type = CONV_TYPE::FULL);

protected:
private:
    imath(const imath &) = delete;
    imath &operator=(const imath &) = delete;

public:
protected:
private:
};

} // namespace nmath

#endif // __SOFTWARE_CONFIG_MATH_H_
