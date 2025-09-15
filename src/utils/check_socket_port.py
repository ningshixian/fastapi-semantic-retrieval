import socket
import time
import os
import subprocess


def _is_available(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # 设置超时时间为1秒
    ex_result = sock.connect_ex((ip, port))  # 返回状态值
    sock.close()
    if ex_result == 0:
        # print("Port %d is open" % port)
        return True
    else:
        print("Port %d is not open" % port)
        return False


while True:
    time.sleep(10)
    ip = "127.0.0.1"  # ip对应服务器的ip地址
    port = 8090

    if _is_available(ip, port):
        loader = subprocess.Popen(
            [
                "nohup",
                "gunicorn",
                "main:app",
                "-b", "0.0.0.0:9000",
                "-w", "1",
                "--threads", "100",
                "-k", "uvicorn.workers.UvicornWorker",
                "> logs/main.log 2>&1 &"
            ]
        )
        returncode = loader.wait()  # 阻塞直至子进程完成
        # print("returncode= %s" %(returncode)) ###打印子进程的返回码
