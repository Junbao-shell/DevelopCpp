[TOC] 

# Target

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

