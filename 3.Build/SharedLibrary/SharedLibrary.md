# 动态库的生成
## Linux

# 动态库的使用
动态库的使用有两种方式，一种是 Linking 的模式，另一种是 Loading 的模式
通过linking 的方式加载动态库，需要在编译时检查动态库
通过loading的方式加载动态库，即在运行时可以选择加载动态库，如果运行时不需要，即不加载
## Liking
+ `target_link_library(shared_library_name)` 链接动态库
+ `#include "header.h" ` 包含头文件，调用接口函数

## Loading
+ `#include <dlfcn.h>`
+ `auto handle = dlopen("libname.so", mode) // mode: enum RTLD_LAZY RTLD_NOW` 
+ `dlsym(handle, "function_name")`

# 静态库
