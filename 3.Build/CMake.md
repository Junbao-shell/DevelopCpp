[toc]

# Base

## 变量

`set()` 的使用

定义环境变量，定义普通变量，定义缓存变量
```cmake
# 声明一个环境变量
set(ENV {<variable>} <value>...)
# 声明一个普通变量
set(<variable> <value>... [PARENT_SCOPE]) # PARENT_SCOPE 修改的时候使用
# 声明一个缓存变量
set(<variable> <value>... CACHE <type> <dicstring> [FORCE])
```
### Normal Variables

```cmake
# 声明一个 normal 变量
set(<variable> <value>... [PARENT_SCOPE]) # PARENT_SCOPE 修改的时候使用
```

`normal` 变量的作用域为整个 CMakeLists.txt

在 `add_subdirectory()` and `function()` 时，如果直接使用 `set(<variable> <value>...)` 只能在子作用域对变量的值进行拷贝，但是不能修改变量的值

```cmake
function(normal_variable var1)
	message(STATUS "var in function: ${var1}")
	set(var1 "var_in_function" PARENT_SCOPE)
    message(STATUS "var in function: ${var1}")
end_function(normal_variable)

set(var1 "vari_in_main")
message(STATUS "var in main: ${var1}")
normal_Variable(var1)
message(STATUS "var in main: ${var1}")
```

上述脚本执行的结果

<img src="../.picture/3.build/cmake_normal_variable.png">

> add_subdirectory() 和 function() 中使用 set(`<variable>` `<value>`) 是值得拷贝，需要通过 `PARENT_SCOPE` 修改变量的值
>
> include() find_package() macro() 中可以通过 `set` 修改变量的值

### Cache Variables

```cmake
# 声明一个 cache 变量
set(<variable> <value>... CACHE <type> <dicstring> [FORCE])
```

`Cache` 变量相当于一个全局变量，具有以下特点

+ 所有的 `Cache` 变量都会保存在 `CMakeCache.txt` 文件中

  > 由于 `Cache` 变量会保存在 `CMakeCache.txt` 文件中， 所以有时修改变量时，注意可能要删除 `CMakeCache.txt` 才会生效
  >
+ 在 cmake 模块中出现于现有 `Cache` 变量同名的变量值后，以后的 `Cache` 变量使用新的值

  e.g.

  ```cmake
  set(var "888" CACHE STRING INTERNAL)
  message(STATUS "var = ${var}")
  set(var "666")
  message(STATUS "var = ${var}")
  ```

### set 的应用
#### 设置编译选项
通过环境变量设置
```cmake
CMAKE_CXX_FLAGS 是cmake内置的环境变量
set(CMAKE_CXX_FLAGS
    -std=c++11
    -)
```

## Target
通过 target 与构建和使用的所有依赖建立绑定关系
+ target_sources
+ target_include_directories
+ target_compile_definition
+ target_compile_options
+ target_compile_features
+ target_link_options

在软件，算法开发过程中，有很多借口都是软件内部使用，不希望暴露给外部。只暴露外部需要用到的接口。`target` 引入 **user requirement** 和 **compile requirement**，通过 `INTERFACE, PUBLIC, PRIVATE` 标识不同的作用域。

其中：
+ `INTERFACE` 标识添加的头文件路径仅 target 的使用方需要，编译当前 target不需要
+ `PRIVATE` 与 `INTERFACE` 相反
+ `PUBLIC` 标识都需要

### `target_sources`

使用 `target_sources()` 而不是 `file(GLOB/GLOB_RECURSE) `
因为使用 `GLOB` 正则匹配后，增删文件cmake系统无感知。**cmake 官方强烈建议不使用 `GLOB` 的方式引入源文件。** 

### `target_include_directories`

### `target_compile_definition`

### `target_compile_options`

### `target_compile_features`
 
### `target_link_options`

### `get_target_property`

### `set_target_properties`

### `get_property(TARGET)`

### `get_property(TARGET)`


## Install

```cmake
install(TARGET my_lib
        EXPORT my_lib_targets
        LIBRARY DESTINITION lib
        ARCHIVE DESTINITION lib
        RUNTIME DESTINITION bin
        PUBLIC_HEADER DESTINITION include
)
```


### 尝试一下新的键盘

```cmake
set(CMAKE_CXX_FLAGS "-Wall")

```
明天以及下一工作周的计划
+ cmake 时间，掌握最基本的适合C++的CMakeLists.txt的模板
+ 熟悉10个小的开源项目的 CMakeLists.txt 
+ 使用python编写自动构建项目的脚本
+ 编写 .cmake模块

继续熟悉python, 主要基本语法
编写文件查重的脚本，检查两个文件夹下的md5值，比较文件夹的异同和文件的异同

shell脚本，目前的开发环境基于linux, 需要熟悉linux shell脚本的基本语言

CUDA

数字图像处理，基本的图像处理算法，图像降噪算法





