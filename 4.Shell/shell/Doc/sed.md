[TOC]

# Sed 

## Introduction
`sed` 是 Linux 利用脚本处理文本文件的一个工具，用来自动编辑一个或多个文件，简化对文件的反复操作，编写转换程序；
`sed` 主要是面向行进行处理的工具，每次只处理一行的内容；

`sed` 的工作流程：将 **要处理的行** 输入到缓冲区, `sed` 处理缓冲区的内容, 处理完成后将缓冲区的内容打印到屏幕上。所以 `sed` 默认不会更改文本文件的内容。

```shell
sed [-hnV][-e<script>][-f<script文件>][文本文件]
```
参数解析：
```shell
+ -h  # --help 
+ -n  # --quite --slient 显示script处理后的结果 
+ -V  # --version 显示版本信息 
+ -e <script> # --expression=<script> 以选项中指定的 script 来处理输入的文本文件
+ -f <script> # --file=<script> 以选项中指定的 script 来处理输入的文本文件
```

## action command
sed 后面直接连接动作 需要用使用一对单引号括注
例如
```shell
sed `2,4d`
```

### `c` 替换命令
```shell
sed "s/search_string/replace_string/g" target_file
```
当前命令行中的分隔符