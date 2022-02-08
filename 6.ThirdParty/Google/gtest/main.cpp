///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief UT google test module 
/// 
/// @file main.cpp
/// 
/// @author Jonathan
/// 
/// @date 2022-02-08
///////////////////////////////////////////////////////////

// Current Cpp header
#include "test.h"
// System header
// C/C++ standard library header
#include <iostream>
// External library header
#include <gtest/gtest.h>
// Current module header
// Root directory header

int main(int argc, char **argv)
{
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);

    auto ret = RUN_ALL_TESTS();
    if (!ret)
    {
        LOG(WARNING) << "run all test return warning value";
    }

    google::ShutdownGoogleLogging();
    return 0;
}

