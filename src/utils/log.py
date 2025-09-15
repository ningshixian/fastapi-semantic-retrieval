import os
import sys
import time
from functools import wraps
import traceback
import logging
from loguru import logger
from fastapi import status
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder

"""
可以将 log 的配置和使用更加简单和方便
默认的输出格式包括：时间、级别、模块名、行号以及日志信息
pip install loguru
https://cloud.tencent.com/developer/article/2295354
"""

# 定位到log日志文件
log_path = os.path.join('./', 'logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)

# # 彩色格式
# from colorlog import ColoredFormatter
# formatter = ColoredFormatter(
#     "%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - %(blue)s%(message)s",
#     datefmt='%Y-%m-%d %H:%M:%S',
#     reset=True,
#     log_colors={
#         'DEBUG': 'cyan',
#         'INFO': 'green',
#         'WARNING': 'yellow',
#         'ERROR': 'red',
#         'CRITICAL': 'bold_red',
#     },
#     secondary_log_colors={},
#     style='%'
# )
# # --- 日志配置 ---
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.propagate = False  # 防止日志传播到根logger
# # 控制台输出配置
# sh = logging.StreamHandler(sys.stdout)  #往屏幕上输出
# sh.setFormatter(formatter) #设置屏幕上显示的格式
# # 添加 handler 到 logger
# logger.addHandler(sh)


# logger的完整配置
fmt="{time:YYYY/MM/DD at HH:mm:ss} | {level} | {file}:{line} | {message}"
logger.add(
    sink=os.path.join(log_path, f'{time.strftime("%Y-%m-%d")}_info.log'),
    rotation='200 MB',                  # 日志文件最大限制200mb
    retention='30 days',                # 最长保留30天
    format=fmt,                         # 日志显示格式
    # compression="zip",                  # 压缩形式保存
    encoding='utf-8',                   # 编码
    level='INFO',                       # 日志级别
    filter=lambda record: record["level"].name == "INFO",
    enqueue=True,                       # 将日志消息放入队列中，异步日志记录。日志消息会被后台线程处理，不会阻塞主程序的执行。
    # serialize=True,                   # Loguru 会将全部日志消息转换为 JSON 格式保存
)
fmt="{time:YYYY/MM/DD at HH:mm:ss} | <red>{level}</red> | {file}:{line} | <level>{message}</level>"
logger.add(
    sink=os.path.join(log_path, f'{time.strftime("%Y-%m-%d")}_error.log'),
    rotation='200 MB',                  # 日志文件最大限制200mb
    retention='30 days',                # 最长保留30天
    format=fmt,                        # 日志显示格式
    # compression="zip",                  # 压缩形式保存
    encoding='utf-8',                   # 编码
    level='ERROR',                       # 日志级别
    filter=lambda record: record["level"].name == "ERROR",
    enqueue=True,                       # 将日志消息放入队列中，异步日志记录。日志消息会被后台线程处理，不会阻塞主程序的执行。
    # serialize=True,                   # Loguru 会将全部日志消息转换为 JSON 格式保存
)


# logger.info("logger init")
# logger.error("logger error")

# # 开启了enqueue（队列模式）后，如何关闭后台日志记录
# logger.complete()  # 等待队列中的消息处理完
# logger.remove()    # 移除 handler
# exit()


"""logger 使用方式1
Traceback 记录，使用 Loguru 提供的装饰器进行 Traceback 的记录
"""

# Loguru 支持使用装饰器的方式捕获方法可能出现的异常
@logger.catch
def risky_function():
    return 1 / 0  # 这会引发一个异常


"""logger 使用方式2
装饰器来捕获代码异常&记录日志
"""

def log_filter(func):
    """装饰器来捕获代码异常&记录日志"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = 1000 * time.time()
        logger.info(f"=============  Begin: {func.__name__}  =============")
        logger.info(f"Request: {kwargs['request'].url}")
        logger.info(f"Args: {kwargs['item2'] if 'item2' in kwargs else kwargs['item']}")
        try:
            rsp = func(*args, **kwargs)
            logger.info(f"Response: {rsp.body.decode('utf-8')}") 
            end = 1000 * time.time()
            logger.info(f"Time consuming: {end - start}ms")
            logger.info(f"=============   End: {func.__name__}   =============\n")
            return rsp
        except Exception as e:
            logger.error(traceback.format_exc())  # 错误日志 repr(e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=jsonable_encoder({
                    'code': 500,
                    'message': "Internal Server Error",
                    'data': {
                        "text": "",
                        "origin_text": "",
                        "answer": "",   # 传给DM用于前端展示
                        "answer_type": -1,   # top1回复类型 0：空 1：精准匹配 3：模糊匹配
                        "confidence": -1,  # top1 score
                        "threshold": {"high": 0.88, "low": 0.44},  # 高/低阈值
                        "detail_results": [],  # 向量检索的所有结果
                    },
                })
            )
    return wrapper


# @log_filter
# def main():
#     print("ceshi")


__all__ = ["logger", "log_filter"]