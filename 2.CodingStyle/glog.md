## 
+ init google log system
  google::InitGoogleLogging(argv[0])
+ shutdown
  google::ShutdownGoogleLogging();

## 日志级别
+ INFO 
+ WARNING 
+ ERROR 
+ FATAL

## 常用的FLAG
+ FLAGS_log_dir 设置日志输出路径 通过 `google::ParseCommandLineFlags(&argc, &argv, true)`
    <project_name>.<host_name>.<user_name>.log.<severity_level>.<date>.<time>.<pid>
+ FLAGS_v 自定义 `VLOG(level)` 的日志等级, 小于等于level值才会输出
+ FLAGS_max_log_size 每个日志文件的最大大小（unit: MB）
+ FLAGS_minloglevel 输出日志的最小级别
+ 
+ 
## LOG
+ LOG(INFO)
+ LOG(WARNING)
+ LOG(ERROR)
+ LOG(INFO, condition) // LOG(INFO, a > 0) 当 a > 0 的时候输出日志
+ LOG_FIRST_N(INFO, N) // LOG_FIRST_N(INFO, 10) 此代码执行前10次打印，超过10次后不打印
+ LOG_EVERY_N(INFO, N) // LOG_EVERY_N(INFO, 10) 每10次打印一次
+ LOG_IF_EVERY_N(INFO, N) // LOG_IF_EVERY_N(INFO, condition, 10) 符合条件condition, 每10次执行一次
+ 
# DLOG Debug log // only open in debug model
+ DLOG(INFO)
+ DLOG(WARNING)
+ DLOG(ERROR)
+ DLOG_IF()
+ DLOG_EVERY_N()
+ 
# Verbose LOG
+ VLOG(level)
+ 
+ 
+ 













