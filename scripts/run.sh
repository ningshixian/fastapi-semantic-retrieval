#!/bin/bash

export FAQ_ENV=prod     # 线上生产环境
export CUDA_VISIBLE_DEVICES=3


# ==================================================================== #
# # FAQ服务环境搭建
# # 仅在第一次部署 or 服务迁移时使用！
# ==================================================================== #

# 安装必要的软件
apt-get update
apt-get install lsof

# 如果是第一次部署服务 or 服务迁移，clone 最新项目代码
git clone https://xxx.git

mkdir data_factory/kafka
mkdir data_factory/faq
# 需 new 两个数据文件
touch data_factory/faq/qa_qy_onetouch.json
echo "[]" > data_factory/faq/qa_qy_onetouch.json
touch data_factory/faq/slot.json
echo "{}" > data_factory/faq/slot.json

# 依赖安装(如果出问题，请先执行上一个命令)
pip install -r requirements.txt         # 针对 cu121 环境
pip install -r requirements-dev.txt       # 针对 cu117 环境
# cu121 还需额外安装依赖包
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --upgrade    # 2.7.4.post1


# ==================================================================== #
# # FAQ服务启动/重启步骤
# ==================================================================== #

conda activate pp # 测试激活，线上无需

# # 拉取最新项目代码
# git pull
# # 拉取最新项目代码(强制)
# git fetch --all
# git reset --hard origin/master

# 验证下（ok就直接退出）
gunicorn application:app -c gunicorn/gunicorn_config.py

# kill掉旧进程
pkill -f "socket8092_detection.py"
pkill -f consumer_qy_v2.py
pkill -f crontab_full_update_v2.py
pkill -f crontab_incremental_update_v2.py
pkill -f "gunicorn application:app -c gunicorn/gunicorn_config.py"

# ❗️必须先手动获取最新知识 - 2025.08.01 新增
python -c "from data_sync.get_crm_knowledge_v2 import *"

# 1、启动主服务
nohup gunicorn application:app -c gunicorn/gunicorn_config.py > logs/app.log 2>&1 &
lsof -i:8092
tail -f logs/app.log 
# 等待服务启动完毕...（Application startup complete.）

# 2、逐一启动辅助脚本

# 启动【服务端口监听】脚本
nohup python -u socket8092_detection.py > logs/socket.log 2>&1 &
ps -ef|grep socket

cd data_sync
cd data_sync

# 启动【知识拉取】定时任务脚本
nohup python -u crontab_full_update_v2.py > ../logs/full_update.log 2>&1 &
ps -ef|grep crontab

# 启动【kafka消息监听】服务
nohup python -u consumer_qy_v2.py > ../logs/consumer.log 2>&1 &
ps -ef|grep consumer


# # 出问题的话回退版本
# git reset --hard HEAD~1    # 回退到上一个版本
# git reset --hard commit_id # 回退到指定版本

# ```
# curl --location --request POST 'localhost:8092/predict4cc' \
# --header 'accept: application/json' \
# --header 'Content-Type: application/json' \
# --data-raw '{
#   "text": "xxx",
#   "top_n": 5,
#   "search_strategy":"hybrid"
# }'
# ```