import sys
import time
import subprocess
import requests
# 引入定时任务库
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
# 引入时间库
import pytz
from zoneinfo import ZoneInfo
from datetime import datetime
# 导入本地库
import sys
sys.path.append(r"../")
from configs.config import urls

"""
全量同步-命令：
cd data_sync
pkill -f crontab_data.py
nohup python crontab_data.py > ../logs/crontab.log 2>&1 &

注意：隔一段时间需要清理cronlog.txt日志文件
"""

# 1. 定义时区
# beijing_tz = pytz.timezone('Asia/Shanghai')
beijing_tz = ZoneInfo('Asia/Shanghai')
print(f"调度器时区: {beijing_tz}")
print(f"北京时间: {datetime.now(beijing_tz)}")
print(f"UTC时间: {datetime.now(pytz.UTC)}")
print(f"系统本地时间: {datetime.now()}")


def get_now_time():
    now = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")
    return now


def scheduled_job():
    print(f"{get_now_time()} 开始执行任务")
    print(get_now_time())
    loader = subprocess.Popen(["python", "combile_data.py"])
    returncode = loader.wait()  # # 等待子进程结束，并获取退出状态码
    if returncode == 0:
        print(f"{get_now_time()} 【知识拉取】成功！\n")
    else:
        print(f"{get_now_time()} 【知识拉取】失败！\n")


def scheduled_job2():
    print("faq向量更新......")
    print(get_now_time())
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", urls.full_update, headers=headers)
    if response.status_code == 200:
        print(f"{get_now_time()} 【faq向量更新】成功！\n")
    else:
        print(f"{get_now_time()} 【faq向量更新】失败！\n")


def dojob():
    # 创建调度器：BlockingScheduler 
    scheduler = BlockingScheduler(timezone=beijing_tz)

    # # 未显式指定，那么则立即执行
    # scheduler.add_job(auto_update_json, args=[])

    # # 添加定时任务，每5分钟执行一次
    # scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('*/5 * * * *'), id='knowledge accquire')
    
    # 添加定时任务，每天凌晨12点 trigger='cron' 
    scheduler.add_job(
        scheduled_job, 
        trigger=CronTrigger.from_crontab('30 0 * * *', timezone=beijing_tz), 
        id='knowledge accquire'
    )
    scheduler.add_job(
        scheduled_job2, 
        trigger=CronTrigger.from_crontab('35 0 * * *', timezone=beijing_tz), 
        id='fqa update'
    )

    # 4. 启动
    scheduler.start()


if __name__ == "__main__":
    dojob()
