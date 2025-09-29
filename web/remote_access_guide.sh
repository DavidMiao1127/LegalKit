#!/bin/bash

# SSH端口转发脚本
# 请根据您的实际情况修改以下参数

# 远程服务器配置
REMOTE_HOST="your-server-ip"           # 替换为您的服务器IP
REMOTE_USER="your-username"            # 替换为您的用户名
REMOTE_PORT=22                         # SSH端口，通常是22

# 端口转发配置
LOCAL_PORT=5000                        # 本地端口
REMOTE_PORT_APP=5000                   # 远程LegalKit Web端口

echo "======= LegalKit Web 远程访问配置 ======="
echo ""
echo "如果您无法直接访问服务器IP，请使用以下方法之一："
echo ""

echo "方法1: SSH端口转发命令"
echo "在您的本地电脑上运行以下命令："
echo ""
echo "ssh -L ${LOCAL_PORT}:localhost:${REMOTE_PORT_APP} ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo "然后在本地浏览器访问: http://localhost:${LOCAL_PORT}"
echo ""

echo "方法2: VS Code端口转发（如果使用VS Code Remote）"
echo "1. 在VS Code中，按 Ctrl+Shift+P"
echo "2. 搜索并选择 'Ports: Focus on Ports View'"
echo "3. 点击 'Forward a Port'"
echo "4. 输入端口号: ${REMOTE_PORT_APP}"
echo "5. VS Code会自动生成访问链接"
echo ""

echo "方法3: 直接访问（如果网络允许）"
echo "从启动日志可以看到，服务运行在以下地址："
echo "- http://127.0.0.1:5000 （服务器本地）"
echo "- http://10.254.0.8:5000 （内网IP）"
echo ""

echo "选择适合您网络环境的方法访问LegalKit Web界面"
echo "============================================="