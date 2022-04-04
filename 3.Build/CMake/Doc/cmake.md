[TOC]

# 

## 项目配置描述
在构建CMakeLists.txt 时，除了向目标为提供源文件，头文件和链接文件，还要提供 **项目配置描述**


## cmake 生成文件
### Linux
在Linux操作系统中默认使用 GNU/Makefile 构建工具，执行 `cmake` 指令将生成如下的文件
执行 `cmake --build CMakeLists.txt` 后生成以下几个文件：
+ Makefile
+ CMakefile
+ cmake_install.cmake # 处理安装规则的脚本，在项目安装时需要
+ CMakeCache.txt


