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

#ifndef __SOFTWARE-CONFIG_CONV_H_
#define __SOFTWARE-CONFIG_CONV_H_

// System header
// C/C++ standard library header
#include <iostream>
// External library header
// Current module header
#include "math.h"
// Root directory header

class Conv : public imath
{
public:
    Conv();
    Conv(const Conv &) = delete;
    Conv &operator=(const Conv &) = delete;
    ~Conv();

    bool conv(const float *, const float *, float *); 

protected:
private:



public:
protected:
private:
};

#endif // __SOFTWARE-CONFIG_CONV_H_
