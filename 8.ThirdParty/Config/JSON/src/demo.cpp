///////////////////////////////////////////////////////////
/// @copyright copyright description
///
/// @brief test json config module, the first json demo
///
/// @file demo.cpp
///
/// @author Jonathan
///
/// @date 2022-02-21
///////////////////////////////////////////////////////////

// Current Cpp header
// System header
// C/C++ standard library header
// External library header
#include "json.hpp"
#include <glog/logging.h>
// Current module header
// Root directory header

int main(int argc, char **argv)
{
    FLAGS_v = 4;
    FLAGS_log_dir = "./log";
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);

    

    google::ShutdownGoogleLogging();

    return 0;
}
