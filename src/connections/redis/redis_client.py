import os
import json
import redis
from datetime import timedelta
from typing import Optional, Union, Any
from redis import Redis, ConnectionPool
from redis import asyncio as aioredis
from redis.sentinel import Sentinel


class RedisConfig(object):
    """
    Redis 配置类（普通模式）
    """
    host = "127.0.0.1"          # Redis 服务器地址
    port = 6379                 # Redis 端口
    username = None             # 用户名（可选）
    password = "******"         # 密码（如无需可设为 None）
    database = 0                # 数据库索引
    max_connections = 100       # 最大连接数


class SentinelConfig(object):
    """
    Redis Sentinel 配置类
    """
    redis_sentinel_list = ["127.0.0.1:26379"]  # Sentinel 节点列表
    sentinel_master: str = "mymaster"          # 主节点名称
    redis_password: str = "******"             # Redis 连接密码


class UnifiedRedisUtils(object):
    """
    统一的 Redis 工具类，支持普通连接和 Sentinel 模式
    """

    def __init__(self, config: RedisConfig = None, sentinel_config: SentinelConfig = None):
        """
        初始化 Redis 客户端
        Args:
            config: RedisConfig，用于普通连接
            sentinel_config: SentinelConfig，用于高可用模式
        """
        self.redis_pool = None
        self.redis_connection = None
        self.sentinel = None
        self.master = None
        self.slave = None
        
        if sentinel_config:
            self._init_sentinel(sentinel_config)
        else:
            self._init_redis(config or RedisConfig())
    
    def _init_redis(self, config: RedisConfig):
        """初始化普通 Redis 连接"""
        self.redis_pool = ConnectionPool(
            host=config.host,
            port=config.port,
            max_connections=config.max_connections,
            username=config.username,
            password=config.password,
            db=config.database,
        )
        self.redis_connection = Redis(connection_pool=self.redis_pool)
    
    def _init_sentinel(self, sentinel_config: SentinelConfig):
        """初始化 Redis Sentinel 连接"""
        sentinel_list = []
        for sentinel_ip in sentinel_config.redis_sentinel_list:
            ip, port = sentinel_ip.split(":")
            sentinel_list.append((ip, int(port)))

        self.sentinel = Sentinel(
            sentinel_list,
            min_other_sentinels=0,
            sentinel_kwargs={
                "password": sentinel_config.redis_password
            },
        )

        self.master = self.sentinel.master_for(
            sentinel_config.sentinel_master,
            db=0,
            password=sentinel_config.redis_password,
            decode_responses=True,
        )

        self.slave = self.sentinel.slave_for(
            sentinel_config.sentinel_master,
            db=0,
            password=sentinel_config.redis_password,
            decode_responses=True,
        )

        # 简单的可用性检测
        self.master.ping()
        self.slave.ping()
    
    def get_redis_client(self):
        """返回 Redis 客户端"""
        if self.sentinel:
            return self
        return self.redis_connection
    
    def get(self, key: str) -> Any:
        """获取键对应的值（自动解析 JSON）"""
        if self.sentinel:
            val = self.slave.get(key)
            return json.loads(val) if val else None
        else:
            value = self.redis_connection.get(key)
            if value:
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
    
    def set(self, key: str, val: Any, ex: Optional[int] = 2*24*60*60) -> bool:
        """设置键值（默认过期 2 天）"""
        if self.sentinel:
            return self.master.set(
                key, json.dumps(val, ensure_ascii=False), ex=ex
            )
        else:
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            return self.redis_connection.set(key, val, ex=ex)
    
    def delete(self, key: str) -> int:
        """删除指定键"""
        if self.sentinel:
            return self.master.delete(key)
        else:
            return self.redis_connection.delete(key)
    
    def set_string(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """直接设置字符串值"""
        if self.sentinel:
            return self.master.set(key, value, ex=ex)
        else:
            return self.redis_connection.set(key, value, ex=ex)
    
    def get_string(self, key: str) -> Optional[str]:
        """获取字符串值"""
        if self.sentinel:
            value = self.slave.get(key)
        else:
            value = self.redis_connection.get(key)
            
        if value:
            return value.decode('utf-8') if isinstance(value, bytes) else value
        return None
    
    def expire(self, key: str, ex: int) -> bool:
        """设置键的过期时间"""
        if self.sentinel:
            return self.master.expire(key, ex)
        else:
            return self.redis_connection.expire(key, ex)


# 使用示例
if __name__ == "__main__":
    # 常规模式
    redis_utils = UnifiedRedisUtils()
    redis_utils.set('test_key', {'name': 'example', 'value': 123})
    result = redis_utils.get('test_key')
    print(result)
    
    # Sentinel 模式示例（需实际 Sentinel 环境）
    # sentinel_config = SentinelConfig()
    # sentinel_utils = UnifiedRedisUtils(sentinel_config=sentinel_config)
    # sentinel_utils.set('sentinel_key', 'sentinel_value')
    # result = sentinel_utils.get('sentinel_key')
    # print(result)