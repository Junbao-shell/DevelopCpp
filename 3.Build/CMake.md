[TOC]

# Base

## 变量

`set()` 的使用

```cmake
set(ENV {<variable>} <value>...)
```



### Normal Variables

```cmake
# 声明一个 normal 变量
set(<variable> <value>... [PARENT_SCOPE]) # PARENT_SCOPE 修改的时候使用
```

`normal` 变量的作用域为整个 CMakeLists.txt 

在`add_subdirectory()` and `function()` 时，如果直接使用 `set(<variable> <value>...)` 只能在子作用域对变量的值进行拷贝，但是不能修改变量的值

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

<img src="../.picture/3.build/cmake_normal_variable.png" align=left>



> add_subdirectory() 和 function() 中使用 set(<variable> <value>) 是值得拷贝，需要通过 `PARENT_SCOPE` 修改变量的值
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

+ 在 cmake 模块中出现于现有 `Cache` 变量同名的变量值后，以后的 `Cache` 变量使用新的值

  e.g. 

  ```cmake
  set(var "888" CACHE STRING INTERNAL)
  message(STATUS "var = ${var}")
  set(var "666")
  message(STATUS "var = ${var}")
  ```

  