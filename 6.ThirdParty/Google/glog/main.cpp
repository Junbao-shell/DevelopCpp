#include <glog/logging.h>

int main(int argc, char** argv) 
{
    google::InitGoogleLogging(argv[0]);

    // FLAGS_logtostderr = true;
    FLAGS_alsologtostderr = true;
    FLAGS_log_dir = "./";
    FLAGS_v = 2;

    VLOG(0) << " i am 0";
    VLOG(1) << " i am 1";
    VLOG(2) << " i am 2";
    VLOG(3) << " i am 3";

    google::ShutdownGoogleLogging();

    return 0;
}