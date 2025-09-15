import sys
import time
import pytz
from datetime import datetime
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import subprocess
from pytz import timezone

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

def get_now_time():
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")
    return now


def scheduled_job():
    print("获取【知识平台】QA数据对（通用&机器人），定时任务启动!")
    print(get_now_time())
    loader = subprocess.Popen(["python", "combile_data.py"])
    returncode = loader.wait()  # # 等待子进程结束，并获取退出状态码
    if returncode == 0:
        print("【知识拉取】成功！")
    else:
        print("【知识拉取】失败！")
    print(get_now_time())


def scheduled_job2():
    print("faq向量更新......")
    print(get_now_time())
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", urls.full_update, headers=headers)
    if response.status_code == 200:
        print("【faq向量更新】成功！")
    else:
        print("【faq向量更新】失败！")
    print(get_now_time())


def scheduled_job3():
    print("recommend向量更新......")
    print(get_now_time())
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", urls.full_update_recommend, headers=headers)
    if response.status_code == 200:
        print("【recommend向量更新】完成")
    else:
        print("【recommend向量更新】失败！")
    print(get_now_time())
    

# def auto_update_json():
#     scheduled_job()
#     scheduled_job2()
#     scheduled_job3()


def dojob():

    # 检查当前时区设置
    beijing_tz = pytz.timezone('Asia/Shanghai')
    print(f"北京时间: {datetime.now(beijing_tz)}")
    print(f"UTC时间: {datetime.now(pytz.UTC)}")
    print(f"系统本地时间: {datetime.now()}")

    # 创建调度器：BlockingScheduler 
    scheduler = BlockingScheduler(timezone=beijing_tz)

    # # 未显式指定，那么则立即执行
    # scheduler.add_job(auto_update_json, args=[])

    # # 添加定时任务，每5分钟执行一次
    # scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('*/5 * * * *'), id='knowledge accquire')
    # 添加定时任务，每天凌晨12点 trigger='cron' 
    scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('30 0 * * *'), id='knowledge accquire')
    scheduler.add_job(scheduled_job2, trigger=CronTrigger.from_crontab('35 0 * * *'), id='fqa update')
    scheduler.add_job(scheduled_job3, trigger=CronTrigger.from_crontab('45 0 * * *'), id='recommend update')
    scheduler.start()


if __name__ == "__main__":
    dojob()
