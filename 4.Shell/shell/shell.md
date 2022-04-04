[TOC]

## VIM

#### 激活方向键，删除键等
```shell
# 方法1
echo "set nocompatible" >> ~/.vimrc
echo "set backspace=2" >> ~/.vimrc
# 方法2
sudo vi ~/.vimrc
```
## apt

### apt 与 apt-get 的区别
`apt-get` (Advanced Package Tool) 是一款基于Unix和Linux的程序管理器
`apt` 在2014年逐渐开始使用，并在Ubuntu16.04逐渐取代 `apt-get`，
一个区别：
如果进行 update
`apt update` 与 `apt-get update` 前置除了更新存储库的索引，还告知是否有可用软件以及有多少新版本可用；
有限推荐使用 `apt`

### remove software / library

### apt-cache 查看安装库的版本信息

```shell
sudo apt-cache show libboost-dev
```

### 升级软件
以 git 为例
```shell
sudo apt update  # 更新源
sudo apt install software-properties-common # 安装 PPA 需要的依赖
sudo add-apt-repository ppa:git-core/ppa    # 向 PPA 中添加 git 的软件源
sudo apt-get update
sudo apt-get install git
```

### 卸载软件
sudo apt remove 
sudo apt remove --purge 

## ssh
使用 ssh 远程连接服务器，使用一段时间后，远程主机的ssh-key秘钥发生变化，如重新安装系统了等，但是IP地址没有变
这是再使用ssh连接会显示如下的问题
<img src='../../.picture/4.Shell/ssh_remote_host_changed.png'>
<p><center>远程主机发生变化时，重新ssh远程显示报错</center></p>

此时需要使用
```shell
ssh-keygen -f "~/.ssh/known_hosts" -R "xxx.xxx.xxx.xxx" # ip address
```

在windows中重新连接出现问题是，提示 known-host文件的问题
注意在 windows系统盘，用户文件夹 .ssh/konwn_hosts 文件，删除该文件下有问题的远程配置
