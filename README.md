[TOC]

个人简介：编者从事医学影像系统服务端开发方向，主要开发语言为C++, 面向医学影像设备(当前主要涉及PET, CT, 未来还会扩展到MR, UtralSound)进行原始数据处理，图像重建，重建图像的后处理等工作，工作内容主要涉及图像算法开发，例如降噪算法，数据校正算法，图像重建算法，还会涉及到客户端软件界面开发，客户端与服务端的通信开发，设备与服务端的通信功能等。
工作环境：
+ 基于Linux操作系统
+ 基于VS Code编辑器
+ 参考Google的C++编码规范
+ 基于CMake构建工具
+ 使用GTest作为单元测试工具
+ 使用GLOG作为日志工具

本仓库主要是对日常工作和学习过程中相关的内容进行总结和归纳，逐渐形成一个规范的开发体系，养成高效的开发习惯。当前处于前期完善阶段，但STATUS set to PUBLIC, 督促不断完善。
```git
git clone https://gitee.com/junbao-shell/software-config.git
```

Content:
1. 常用软件配置的配置文件，目前主要使用 VS Code
2. CodingStyle, 编码规范，参考Google C++ 指定本项目，以及其他所有项目的编码规范
3. Build, 主要介绍软件构建的基本工具及其对应的使用方法，目前主要使用CMake
4. Shell，日常工作总使用脚本工具，主要为 linux bash shell, python, windows bat偶有涉及
5. Design mode 日常工作中主要使用的涉及模式，目前经常使用为 单例模式和工厂模式
6. 第三方库使用，对工作中使用频率较高的第三方库的使用进行总结，主要涉及 库源码的下载，编译和使用。简单介绍基本内容，使用语法等。如： google glog, gtest, boost等


