# Just是一个简单而强大的命令运行器，它允许你将项目中常用的命令保存在一个名为justfile的文件中。
# justfile是Just的核心，它包含了所有可运行的命令（也称为配方），使用 `just --list` 查看可用任务
# 说明：
# - 默认项目目录：faq-semantic-retrieval-new2
# - 默认服务端口：8092
# - 默认环境变量：
#     export FAQ_ENV=prod     # 线上生产环境
#     export CUDA_VISIBLE_DEVICES=3
#     conda activate pp # 测试激活，线上无需
# - 部分命令需要 root 权限（如 apt-get、lsof、just安装）

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set export := true

# 基本配置
REPO := "faq-semantic-retrieval-new2"
PORT := "8092"
CMD_GUNICORN := "gunicorn application:app -c gunicorn/gunicorn_conf.py"
LOG_DIR := "logs"

# 环境变量（可通过环境覆盖）
FAQ_ENV := env_var("FAQ_ENV")
CUDA_VISIBLE_DEVICES := env_var("CUDA_VISIBLE_DEVICES")

# ================================ #
# 帮助
# ================================ #
help:
	@echo "可用任务："
	@echo "  help                          # 查看帮助"
	@echo "  env                           # 查看当前环境变量"
	@echo "  initial-setup                 # 首次环境搭建（安装 lsof、clone、初始化数据文件）"
	@echo "  fix-pip                       # 修复并设置公司 Artifactory 源"
	@echo "  install-deps-cu121            # 安装依赖（CUDA 12.1 环境）"
	@echo "  install-deps-cu117            # 安装依赖（CUDA 11.7 环境）"
	@echo "  validate                      # 前台验证服务是否可启动"
	@echo "  stop                          # 停止旧进程"
	@echo "  get-latest-knowledge          # 手动获取最新知识（必须先执行）"
	@echo "  start-app                     # 启动主服务（gunicorn）"
	@echo "  start-socket-detection        # 启动服务端口监听脚本"
	@echo "  start-cron-jobs               # 启动知识拉取定时任务脚本"
	@echo "  start-consumer                # 启动 kafka 消息监听服务"
	@echo "  start-all                     # 一键启动（stop -> get-knowledge -> app -> socket -> cron -> consumer）"
	@echo "  tail-app                      # 跟踪主服务日志"
	@echo "  tail-incremental              # 跟踪增量更新日志"
	@echo "  ps-socket                     # 查看 socket 监听进程"
	@echo "  ps-crontab                    # 查看定时任务进程"
	@echo "  ps-consumer                   # 查看 kafka 消费进程"
	@echo "  check                         # 启动后简单自检"
	@echo "  git-pull                      # 拉取最新代码"
	@echo "  git-update                    # 强制拉取最新代码（fetch + reset --hard）"
	@echo "  rollback [commit=HEAD~1]      # 代码回退"
	@echo "  curl-test                     # 本地调用接口示例"
	@echo "  sync-kafka-data [src=../faq-semantic-retrieval/data_factory/kafka_prod]  # 一次性同步旧环境 Kafka 数据并推送 Redis"
	@echo "  model-upload-example          # 模型上传命令示例"
	@echo "  apply-inference-service       # 推理服务申请链接"

# 查看环境变量
env:
	@echo "FAQ_ENV={{FAQ_ENV}}"
	@echo "CUDA_VISIBLE_DEVICES={{CUDA_VISIBLE_DEVICES}}"
	@echo "PORT={{PORT}}"
	@echo "REPO={{REPO}}"

default:
	just --list

# ================================ #
# 首次环境搭建
# ================================ #
initial-setup:
	# 安装必要软件
	sudo apt-get update
	sudo apt-get install -y lsof
	# clone 项目
	if [ ! -d "{{REPO}}" ]; then \
	  git clone https://xxx.git "{{REPO}}"; \
	fi
	# 初始化数据目录与文件
	cd "{{REPO}}" && \
	mkdir -p data_factory/kafka data_factory/faq && \
	if [ ! -f data_factory/faq/qa_qy_onetouch.json ]; then echo "[]" > data_factory/faq/qa_qy_onetouch.json; fi && \
	if [ ! -f data_factory/faq/slot.json ]; then echo "{}" > data_factory/faq/slot.json; fi
	# 创建日志目录
	cd "{{REPO}}" && mkdir -p "{{LOG_DIR}}"

# 依赖安装（CUDA 12.1）
install-deps-cu121:
	cd "{{REPO}}" && \
	pip install -r requirements.txt && \
	pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121 && \
	pip install --upgrade flash-attn

# 依赖安装（CUDA 11.7）
install-deps-cu117:
	cd "{{REPO}}" && \
	pip install -r requirements-dev.txt

# 前台验证服务（启动后看到 "Application startup complete." 立即 Ctrl+C 退出）
validate:
	cd "{{REPO}}" && {{CMD_GUNICORN}}

# 停止旧进程
stop:
	-pkill -f "socket8092_detection.py" || true
	-pkill -f "consumer_qy_v2.py" || true
	-pkill -f "crontab_full_update_v2.py" || true
	-pkill -f "gunicorn application:app -c gunicorn/gunicorn_config.py" || true

# 必须先手动获取最新知识
get-latest-knowledge:
	cd "{{REPO}}" && \
	python -c 'from data_sync.get_crm_knowledge_v2 import *'

# 启动主服务
start-app:
	cd "{{REPO}}" && mkdir -p "{{LOG_DIR}}"
	cd "{{REPO}}" && \
	FAQ_ENV="{{FAQ_ENV}}" CUDA_VISIBLE_DEVICES="{{CUDA_VISIBLE_DEVICES}}" \
	nohup {{CMD_GUNICORN}} > "{{LOG_DIR}}/app.log" 2>&1 &
	sleep 1
	lsof -i:{{PORT}} || true
	@echo "主服务已启动。日志: {{REPO}}/{{LOG_DIR}}/app.log"

# 启动服务端口监听脚本
start-socket-detection:
	cd "{{REPO}}" && mkdir -p "{{LOG_DIR}}"
	cd "{{REPO}}" && \
	nohup python -u socket8092_detection.py > "{{LOG_DIR}}/socket.log" 2>&1 &
	@echo "socket 监听已启动。日志: {{REPO}}/{{LOG_DIR}}/socket.log"

# 启动知识拉取定时任务
start-cron-jobs:
	cd "{{REPO}}/data_sync" && \
	nohup python -u crontab_full_update_v2.py > ../"{{LOG_DIR}}"/full_update.log 2>&1 &
	cd "{{REPO}}/data_sync" && \
	nohup python -u crontab_incremental_update_v2.py > ../"{{LOG_DIR}}"/incremental_update.log 2>&1 &
	@echo "定时任务已启动。日志: {{REPO}}/{{LOG_DIR}}/full_update.log, incremental_update.log"

# 启动 kafka 消息监听
start-consumer:
	cd "{{REPO}}/data_sync" && \
	nohup python -u consumer_qy_v2.py > ../"{{LOG_DIR}}"/consumer.log 2>&1 &
	@echo "Kafka 消费服务已启动。日志: {{REPO}}/{{LOG_DIR}}/consumer.log"

# 一键启动
start-all:
	just stop
	just get-latest-knowledge
	just start-app
	just start-socket-detection
	just start-cron-jobs
	just start-consumer


# ================================ #
# 日志与进程辅助
# ================================ #
tail-app:
	tail -f "{{REPO}}/{{LOG_DIR}}/app.log"

tail-incremental:
	tail -f "{{REPO}}/{{LOG_DIR}}/incremental_update.log"

ps-socket:
	ps -ef | grep -E "socket8092_detection.py" | grep -v grep || true

ps-crontab:
	ps -ef | grep -E "crontab_(full|incremental)_update_v2.py" | grep -v grep || true

ps-consumer:
	ps -ef | grep -E "consumer_qy_v2.py" | grep -v grep || true

# ================================ #
# 启动后检查
# ================================ #
check:
	@echo "检查日志建议："
	@echo "  vi {{REPO}}/{{LOG_DIR}}/app.log"
	@echo "  vi {{REPO}}/{{LOG_DIR}}/socket.log"
	@echo "  vi {{REPO}}/{{LOG_DIR}}/full_update.log"
	@echo "  vi {{REPO}}/{{LOG_DIR}}/incremental_update.log"
	@echo "  vi {{REPO}}/{{LOG_DIR}}/consumer.log"
	@echo "检查数据："
	@echo "  ls -lht {{REPO}}/data_factory/"
	@echo "  ls -lht {{REPO}}/data_factory/kafka/"
	@echo "  vi {{REPO}}/data_factory/kafka/slot.json"
	@echo "  vi {{REPO}}/data_factory/faq/qa_qy_onetouch.json"

# ================================ #
# 代码管理
# ================================ #
git-pull:
	cd "{{REPO}}" && git pull

git-update:
	cd "{{REPO}}" && git fetch --all && git reset --hard origin/master

rollback commit="HEAD~1":
	cd "{{REPO}}" && git reset --hard "{{commit}}"

# ================================ #
# 接口测试
# ================================ #
curl-test host="localhost" endpoint="predict4cc":
    curl --location --request POST '{{host}}:{{PORT}}/{{endpoint}}'
    --header 'accept: application/json'
    --header 'Content-Type: application/json'
    --data-binary @- <<-'JSON'
    {
    "text": "xxx",
    "top_n": 5,
    "search_strategy": "hybrid"
    }
