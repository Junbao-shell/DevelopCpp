[TOC]

本文档描述搭建 `GitLab` 服务器

# 基于 VMWare Ubuntu 16.04 安装 gitlab 服务器


## 修改 初次登陆的 gitlab 密码

初次登陆 gitlab 页面需要输入用户名和密码
初始用户名，为管理员的用户名：root

操作步骤如下：
step1 ：切换到相关路径 `cd /opt/gitlab/bin/`
step2 : 进入控制台  `gitlab-rails console` 
step3 : 查询 root 用户账号信息并赋值给 u `u = User.find(1) `
step4 ：设置密码 `u.password='root123456' `
step5 ：确认密码 `u.password_confirmation='root123456' `
step6 : 保存设置 `u.save`
step7 : 退出控制台 `exit`
step8 : 重启GitLab `gitlab-ctl restart`
step9 ：在浏览器中输入 gitlab服务器的 `url`

