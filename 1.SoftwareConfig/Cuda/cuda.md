[TOC]

# cuda-toolkit & nvidia-driver 安装卸载
如果没有安装 nvidia-driver, 可以直接安装 cuda-toolkit, 安装过程中会自动安装相应的驱动，不需要单独安装
但是如果之前已经安装驱动，则不能直接安装新的驱动以及 cuda-toolkit, 需要将原有的驱动卸载干净，再重新安装

**需要禁用 nouveau**

```shell
# 通过以下命令查看是否禁用 nouveau, 一般情况下是禁用的状态
lsmod | grep nouveau
# 如果没有禁用需要通过以下步骤禁用
# 创建 blacklist.conf
sudo vi /etc/modprobe.d/blacklist.conf
# 写入如下内容
blacklist nouveau options nouveau modesets=0

# 在terminal中重新生成 kernel initramfs
sudo update-initramfs -u

# 重启
sudo reboot
```

**cuda和驱动的安装需要卸载 实现图像界面**
不同操作系统的卸载方式存在一定的差别：（欢迎补充）

```shell
# Ubuntu 16.04
# 禁用图形界面
sudo service lightdm stop # 如果提示没有安装 lightdm 则需要先安装 lightdm 
# 开启图形界面
sudo service lightdm start 

# Ubuntu 18.04
# 禁用图形界面
sudo systemctl set-default multi-user.target
# 开启图形界面
sudo systemctl set-default graphical.target

# 在图形界面时切换到图形界面
startx # 注意不要使用 sudo startx 使用 sudo 权限会进入到 root界面
# 设置开机启动自动进入到图形界面
sudo systemctl set-default graphical.target
sudo reboot 
```

## cuda的安装
首先在官网上下载 cuda 的安装包, 下载 `runfile` 格式的安装包
```shell
# 下载到指定的目录
cd ~/Downloads/
wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda_11.6.1_510.47.03_linux.run 

# 修改文件权限，赋予可执行权限
sudo chmod a+x cuda_11.6.1_510.47.03_linux.run
sudo ./cuda_11.6.1_510.47.03_linux.run

```

## cuda 的卸载
```shell
# 使用官方卸载软件卸载
cd /usr/bin/
sudo ./nvidia-uninstall
cd /usr/local/cuda/bin/
sudo ./cuda-uninstaller
# 卸载残留
sudo apt remove --purge cuda*
sudo apt remove --purge nvidia*
sudo apt autoremove
```

