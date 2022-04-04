///////////////////////////////////////////////////////////
/// @copyright copyright description
///
/// @brief add function for shared library
///
/// @file add.h
///
/// @author Jonathan
///
/// @date 2022-02-13
///////////////////////////////////////////////////////////

#ifndef __SOFTWARE_CONFIG_ADD_H_
#define __SOFTWARE_CONFIG_ADD_H_

// System header
// C/C++ standard library header
// External library header
// Current module header
// Root directory header

class Add
{
public:
    Add() {}
    Add(const Add &) = delete;
    Add &operator=(const Add &) = delete;
    ~Add() {}

    int add(int a, int b);

protected:
private:
public:
protected:
private:
    int a;
    int b;
};

#endif // __SOFTWARE-CONFIG_ADD_H_
