import os
import time
import traceback
import pandas as pd
import pymysql


"""
推荐使用连接池来管理db连接。
"""


# 方案1：使用 SQLAlchemy 连接池（推荐）
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy import text


class SQLAlchemyManager(object):
    DB_URL_TEMPLATE = "{protocol}://{user}:{password}@{host}:{port}/{db}"

    def __init__(self, db_url):
        # 创建连接池，复用连接
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,           # 连接池大小
            max_overflow=10,       # 最大溢出连接数
            pool_pre_ping=True,    # 每次使用前检查连接是否有效
            pool_recycle=3600,     # 连接回收时间（秒）
            echo=False
        )
        
    def query_data(self, sql_query, params=None):
        """执行查询"""
        try:
            # 从连接池获取连接，用完自动归还
            with self.engine.connect() as conn:
                df = pd.read_sql(sql_query, conn, params=params)
                self.logger.info(f"查询成功，返回 {len(df)} 条记录")
                return df
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            raise
    
    def run_monitor(self, sql_query, interval=300):
        """定时监控"""
        while True:
            try:
                df = self.query_data(sql_query)
                # 处理数据...
                self.process_data(df)
            except Exception as e:
                self.logger.error(f"监控任务失败: {e}")
            
            time.sleep(interval)  # 等待5分钟

    def process_data(self, df):
        """处理查询结果"""
        # 你的业务逻辑
        pass
    
    def close(self):
        """关闭连接池"""
        self.engine.dispose()


# # 使用示例
# monitor = SQLAlchemyManager("mysql+pymysql://username:password@host:port/db")
# monitor.run_monitor("SELECT * FROM your_table", interval=300)


# 方案2：使用 PyMySQL + DBUtils 连接池
# 参考：《python DbUtils 使用教程》https://cloud.tencent.com/developer/article/1568031
# 参考：《python数据库连接工具DBUtils》https://segmentfault.com/a/1190000017952033
from DBUtils.PooledDB import PooledDB  # 导入线程池对象


class PooledDBConnection(object):
    def __init__(self, DB_CONFIG):
        # 创建连接池
        self.pool = PooledDB(
            creator=pymysql,
            host=DB_CONFIG.get("host"),
            port=int(DB_CONFIG.get("port")),
            user=DB_CONFIG.get("user"),
            password=DB_CONFIG.get("passwd"),
            db=DB_CONFIG.get("db"),

            charset="utf8", 
            mincached=2,        # 连接池中空闲连接的初始数量(0表示不创建初始空闲连接) 
            maxcached=10,       # 连接池中允许的最大空闲连接数(0或None表示无限制)
            maxshared=3,        # 最大共享连接
            maxconnections=0,   # 允许的最大连接数(0或None表示无限制)    5
            blocking=True,      # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
            ping=4,             # 4 = when a query is executed
            # ping=1,             # 检查连接是否可用
        )

    def get_conn(self):
        """
        连接数据库
        :return: conn, cursor
        """
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()  # pymysql.cursors.DictCursor
        except Exception as e:
            traceback.print_exc()
            print("数据库连接失败========, " + str(e))
            exit()
        return conn, cursor

    def reset_conn(self, conn, cursor):
        """
        :param conn: 数据库连接
        :param cursor: 数据库指针
        :return: Null
        """
        try:
            cursor.close()  # 关闭游标
            conn.close()    # 关闭连接（连接会返回到连接池让后续线程继续使用）
        except Exception as err:
            traceback.print_exc()
            raise ("MySQL关闭异常: ", str(err))

    # 执行查询
    def ExecQuery(self, sql, values=None):
        res = ""
        try:
            conn, cursor = self.get_conn()
            cursor.execute(sql, values)  # 防止SQL注入攻击
            res = cursor.fetchall()
            # res = pd.read_sql(sql, conn, params=values)
            self.reset_conn(conn, cursor)
        except Exception as e:
            traceback.print_exc()
            raise Exception("连接或查询失败：" + str(e))
        return res

    # 执行非查询类语句
    def ExecNonQuery(self, sql, values=None):
        flag = False
        # self.pool.ping(reconnect=True)
        try:
            conn, cursor = self.get_conn()
            conn.begin()   # 开始事务
            cursor.execute(sql, values) # 防止SQL注入攻击
            conn.commit()   # 提交事务
            self.reset_conn(conn, cursor)
            flag = True
        except Exception as e:
            traceback.print_exc()
            conn.rollback()   # 回滚事务
            raise Exception("连接或执行失败: " + str(e))
        return flag


# # 读取数据库配置文件 config.ini
# from configparser import RawConfigParser
# CFG = RawConfigParser()
# CFG.read("config_file.ini", encoding="utf-8")
# monitor = PooledDBConnection(CFG)


# 方案3：使用单个长连接


class PyMysqlConnection:
    def __init__(self, **db_config):
        self.db_config = db_config
        self.connection = None
        self.cursor = None

    def __del__(self):
        # 关闭数据库连接
        self.connection.close()

    def connect(self, server):
        """建立连接"""
        try:
            self.connection = pymysql.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("数据库连接成功~")
        except Exception as e:
            print("数据库连接失败！" + str(e))
            exit()
    
    def ensure_connection(self):
        """确保连接有效"""
        try:
            # 使用 ping 检查连接
            self.connection.ping(reconnect=True)
        except:
            self.logger.info("连接已断开，重新连接...")
            self.connect()

    def query_data(self, sql_query, params=None):
        """执行查询"""
        self.ensure_connection()
        try:
            df = pd.read_sql(sql_query, self.connection, params=params)
            self.logger.info(f"查询成功，返回 {len(df)} 条记录")
            return df
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            # 查询失败可能是连接问题，下次会重连
            self.connection = None
            raise
    
    # 执行非查询类语句
    def ExecNonQuery(self, sql, autoclose=False):
        self.ensure_connection()
        flag = False
        try:
            with self.connection.cursor() as cursor:  # 查询游标
                cursor.execute(sql)
            self.connection.commit()
            if autoclose:
                self.close()
            flag = True
        except Exception as err:
            self.connection.rollback()
            print("执行失败, %s" % err)
        return flag

    def close(self):
        if self.connection:
            self.connection.close()
            print("MySQL连接已关闭")


# # 使用示例
# monitor = PyMysqlConnection(
#     host='localhost',
#     user='root',
#     password='password',
#     database='test'
# )
# monitor.run_monitor("SELECT * FROM your_table")
