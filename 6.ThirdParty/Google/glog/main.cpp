///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief test glog demo
/// 
/// @file main.cpp
/// 
/// @author Jonathan
/// 
/// @date 2022-02-21
///////////////////////////////////////////////////////////

// Current Cpp header
// System header
// C/C++ standard library header
#include <iostream>
// External library header
#include "glog/logging.h"
// Current module header
// Root directory header

int main(int argc, char **argv)
{
    FLAGS_v = 4;
    FLAGS_log_dir = "./log";
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);
    
    VLOG(0) << " i am 0";
    VLOG(1) << " i am 1";
    VLOG(2) << " i am 2";
    VLOG(3) << " i am 3";

    google::ShutdownGoogleLogging();

    return 0;
}
