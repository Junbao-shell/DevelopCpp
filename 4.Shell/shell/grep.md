[TOC]

# Introduction
`grep` : global regular expression print 

## 基本语法
```shell
grep [options] regex [file...]
```

## 常用选项
```shell
-i # ignore 忽略大小写
-v # 不匹配，通常用来去除grep本身
-c # count 匹配的数量
-n # 输出行号
-h # 用于多文件搜索，不输出文件名
```

## 正则表达式

### 元字符
```shell
^ $ . [] {} - ? * +  () | \
```
如果想要直接打印元字符，需要使用转义字符。转义字符可以对自己进行转义

# Example

# 附录

