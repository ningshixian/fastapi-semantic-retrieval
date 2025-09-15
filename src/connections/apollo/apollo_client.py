# -*- encoding: utf-8 -*-
"""
Apollo配置管理类
整合了多个Apollo相关功能
"""

import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Union

# 假设这些模块存在，根据原文件中的引用
# from config.LogConfig import logger
# from util.RsaUtil import RsaUtil
# from Args_util import ARGS
# from apollo.apollo_client import ApolloClient
# from common import constants
# from utils.config_utils import ConfigParser

class UnifiedApolloConfig:
    """
    统一的Apollo配置管理类，整合了多种Apollo配置获取和管理功能
    """
    
    def __init__(self, 
                 apollo_config: Dict[str, Any] = None,
                 env: str = "sit",
                 private_key_path: str = "/etc/apollo/apollo_private_key",
                 api_key: str = None,
                 max_workers: int = 20):
        """
        初始化Apollo配置管理器
        
        Args:
            apollo_config: dict，apollo配置，包含{uri，app_id，cluster_name，token，name_space}
            env: str，环境类型，默认为"sit"
            private_key_path: str，apollo私钥文件路径
            api_key: str，API密钥
            max_workers: int，监听线程池最大工作线程数
        """
        # 环境验证
        if env not in ("sit", "prod", "pre"):
            raise KeyError("{} for env".format(env))
            
        self._env = env
        self._api_key = api_key
        self._apollo_config = apollo_config or {}
        self._stop = False
        self._notification_map = {self._apollo_config.get("name_space", "application"): -1}
        self._data = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # 私钥处理
        self._private_key_path = private_key_path
        self._private_key = None
        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, "r") as f:
                self._private_key = f.read()
        
        # 初始化配置
        if apollo_config:
            self._init_from_apollo(apollo_config)
            self.start_listen()
    
    def _init_from_apollo(self, apollo_config: Dict[str, Any]):
        """
        从Apollo服务器初始化配置
        
        Args:
            apollo_config: Apollo配置字典
        """
        try:
            # 构建URL并获取配置
            url = ("{server_url}/configs/{app_id}/{cluster_name}+{token}/"
                   "{name_space}".format(server_url=apollo_config["uri"],
                                         app_id=apollo_config["app_id"],
                                         cluster_name=apollo_config["cluster_name"],
                                         token=apollo_config["token"],
                                         name_space=apollo_config["name_space"]))
            
            res = requests.get(url=url, timeout=10)
            response_data = json.loads(res.text)
            self._data = response_data.get('configurations', {})
            
            # 解密配置
            self._decrypt()
            
        except Exception as e:
            # logger.error(f"Apollo配置初始化失败: {e}")
            raise e
    
    def get_info_from_apollo(self, 
                           config_server_url: str,
                           app_id: str,
                           cluster_name: str,
                           token: str,
                           namespace_name: str,
                           password_key: str = None,
                           decrypt_url: str = None) -> Dict[str, Any]:
        """
        从Apollo获取指定命名空间的配置信息，并可选择解密特定字段
        
        Args:
            config_server_url: Apollo配置服务器URL
            app_id: 应用ID
            cluster_name: 集群名称
            token: 访问令牌
            namespace_name: 命名空间名称
            password_key: 需要解密的密码字段名
            decrypt_url: 解密服务URL
            
        Returns:
            dict: 配置信息字典
        """
        # 获取配置
        url = (
            "{config_server_url}/configfiles/json/{appId}/{clusterName}+{token}/"
            "{namespaceName}".format(
                config_server_url=config_server_url,
                appId=app_id,
                clusterName=cluster_name,
                token=token,
                namespaceName=namespace_name,
            )
        )
        
        res = requests.get(url=url, timeout=10)
        config_info = json.loads(res.text)
        
        # 如果需要解密且提供了相关参数
        if password_key and decrypt_url and password_key in config_info and self._api_key:
            headers = {
                "Content-Type": "application/json",
                "X-Gaia-API-Key": self._api_key,
            }
            
            body = {
                "privateKey": self._private_key,
                "cipherText": [config_info[password_key]],
            }
            
            decrypt_res = requests.post(url=decrypt_url, headers=headers, data=json.dumps(body))
            config_info[password_key] = json.loads(decrypt_res.text)[0]
        
        return config_info
    
    def _decrypt(self):
        """
        解密配置中的加密字段
        """
        # 简化实现，实际应使用RsaUtil类
        # rsa_util = RsaUtil(apollo_private_key_path=self._private_key_path)
        # for key in self._data:
        #     try:
        #         self._data[key] = rsa_util.decrypt_by_private_key(self._data[key])
        #     except Exception:
        #         continue
        pass
    
    def _heart_listen(self):
        """
        心跳监听配置变更
        """
        while not self._stop:
            try:
                notifications = [{
                    'namespaceName': x,
                    'notificationId': self._notification_map[x]
                } for x in self._notification_map]
                
                listen_params = {
                    'appId': self._apollo_config["app_id"],
                    'cluster': self._apollo_config["cluster_name"],
                    'notifications': json.dumps(notifications)
                }
                
                _heart_url = "{server_url}/notifications/v2".format(
                    server_url=self._apollo_config["uri"])
                r = requests.get(url=_heart_url, params=listen_params, timeout=60)
                
                if r.status_code == 200:
                    # logger.info("配置发生变更，尝试重载配置...")
                    configs = r.json()
                    for config in configs:
                        self._notification_map[config["namespaceName"]] = config["notificationId"]
                        self._init_from_apollo(self._apollo_config)
                    # logger.info("配置重载成功!")
                # else:
                    # logger.error("配置变更发生错误，响应消息为：{}".format(r.text))
                    
            except requests.exceptions.ReadTimeout:
                pass
            except Exception as e:
                # logger.error(f"监听配置变更时发生错误: {e}")
                pass
    
    def start_listen(self):
        """
        启动配置变更监听
        """
        self._thread_pool.submit(self._heart_listen)
    
    def stop_listen(self):
        """
        停止配置变更监听
        """
        self._stop = True
    
    def get_env(self) -> str:
        """
        获取当前环境
        
        Returns:
            str: 当前环境
        """
        return self._env
    
    def get_data(self) -> Dict[str, Any]:
        """
        获取所有配置数据
        
        Returns:
            dict: 配置数据字典
        """
        return self._data.copy()
    
    def get(self, key: str, fallback: Any = None) -> Any:
        """
        获取指定配置项的值
        
        Args:
            key: 配置项键名
            fallback: 默认值
            
        Returns:
            配置项的值或默认值
        """
        return self._data.get(key, fallback)
    
    def getint(self, key: str, fallback: Any = None) -> int:
        """
        获取指定配置项的整数值
        
        Args:
            key: 配置项键名
            fallback: 默认值
            
        Returns:
            int: 配置项的整数值
        """
        try:
            return int(self.get(key, fallback))
        except (ValueError, TypeError):
            return fallback if fallback is not None else 0
    
    def items(self):
        """
        获取所有配置项
        
        Returns:
            list: 配置项列表
        """
        return [(x, self._data[x]) for x in self._data]
    
    @classmethod
    def from_config_parser(cls, 
                          config_parser,
                          env: str = "sit",
                          apollo_keywords_str: str = "apollo_config",
                          private_key_path: str = "/etc/apollo/apollo_private_key") -> 'UnifiedApolloConfig':
        """
        从配置解析器创建Apollo配置管理器
        
        Args:
            config_parser: 配置解析器实例
            env: 环境类型
            apollo_keywords_str: Apollo配置关键字
            private_key_path: 私钥文件路径
            
        Returns:
            UnifiedApolloConfig: Apollo配置管理器实例
        """
        apollo_section_keywords = env + '_' + apollo_keywords_str
        
        apollo_config = {
            "uri": config_parser.get(apollo_section_keywords, "url"),
            "app_id": config_parser.get(apollo_section_keywords, "app_id"),
            "cluster_name": config_parser.get(apollo_section_keywords, "cluster_name", fallback="default"),
            "token": config_parser.get(apollo_section_keywords, "token"),
            "name_space": config_parser.get(apollo_section_keywords, "name_space", fallback="application")
        }
        
        return cls(
            apollo_config=apollo_config,
            env=env,
            private_key_path=private_key_path
        )

# 使用示例
if __name__ == "__main__":
    # 示例1: 直接初始化
    # apollo_manager = UnifiedApolloConfig(
    #     apollo_config={
    #         "uri": "http://localhost:8080",
    #         "app_id": "test-app",
    #         "cluster_name": "default",
    #         "token": "test-token",
    #         "name_space": "application"
    #     },
    #     env="sit"
    # )
    # 
    # # 获取配置
    # port = apollo_manager.getint("SERVER.port", 8080)
    # print(f"Server port: {port}")
    # 
    # # 获取所有配置
    # all_configs = apollo_manager.get_data()
    # print(f"All configs: {all_configs}")
    pass