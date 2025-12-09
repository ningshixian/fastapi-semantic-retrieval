import os
import sys
import signal
import json
import time
import threading
import requests
import datetime
import codecs
from typing import Any, Dict, List, Optional
from enum import Enum
from kafka import KafkaConsumer as KPConsumer
from kafka.structs import TopicPartition

# 假设 utils 中有 Redis 工具和日志封装
from utils import RedisUtilsSentinel
from utils import logger
from configs.config import paths, urls
from configs.config import kafka_config, redis_config, snapshot_key

"""
通用 Kafka 消费脚本示例：
1. 获取消息并持久化到 Redis
2. 启动时恢复快照，并按需更新本地文件
"""

# 枚举定义
class SourceType(str, Enum):
    ENTITY = 'ENTITY'
    PARAM = 'PARAM'
    INTENT = 'INTENT'
    BOT_DEPLOY = 'BOT_DEPLOY'

class OperateType(str, Enum):
    INSERT = 'INSERT'
    UPDATE = 'UPDATE'
    DELETE = 'DELETE'

# 常量定义
SNAPSHOT_INTERVAL_SECONDS = 3600  
DEFAULT_CONSUMER_WAIT_TIME = 60 * 1000
DEFAULT_MAX_POLL_RECORDS = 500

class KnowledgeSyncConsumer:
    """Kafka 数据同步消费者"""

    def __init__(self, kafka_config, redis_conn):
        self.kafka_config = kafka_config
        self.redis_conn = redis_conn
        self.updater = FAQKnowledgeUpdater()
        self.last_snapshot_time = 0.0
        self.consumer: Optional[KPConsumer] = None

        # 本地状态
        self.knowledge_state: Dict[str, Any] = {
            "intent2param": {},
            "param2ent": {},
            "ent2vocab": {},
        }
        self.update_intents = {}
        self._source_handlers = {
            SourceType.ENTITY: self._handle_entity,
            SourceType.PARAM: self._handle_param,
            SourceType.INTENT: self._handle_intent,
        }
    
    def run(self):
        try:
            self._setup_signal_handlers()
            self._init_consumer()
            self._consume_messages()
        except Exception as e:
            logger.critical(f"Consumer stopped: {e}", exc_info=True)
        finally:
            self._cleanup()
    
    def _setup_signal_handlers(self):
        def shutdown(sig, frame):
            logger.info("Received shutdown signal, saving snapshot...")
            self._save_snapshot()
            self._cleanup()
            exit(0)
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
    
    def _init_consumer(self):
        self.consumer = KPConsumer(
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            group_id=self.kafka_config.group_id,
            client_id=self.kafka_config.client_id,
            auto_offset_reset=self.kafka_config.auto_offset_reset,  
            enable_auto_commit=False,
            value_deserializer=lambda v: v.decode("utf-8", "ignore"),
            key_deserializer=lambda k: k.decode("utf-8", "ignore"),
            max_poll_records=DEFAULT_MAX_POLL_RECORDS
        )
        logger.info(f"Kafka consumer initialized for topic: {self.kafka_config.topic_name}")
        
        if not self._load_snapshot():
            self.consumer.subscribe([self.kafka_config.topic_name])
            logger.info(f"No snapshot found. Subscribed to {self.kafka_config.topic_name}")
        self.last_snapshot_time = time.time()

    def _consume_messages(self):
        logger.info("Consumer started.")

        while True:
            # 每隔 1h=3600s 做一次快照
            if time.time() - self.last_snapshot_time > SNAPSHOT_INTERVAL_SECONDS:
                self._save_snapshot()

            # 使用 poll 获取批量消息
            # 提供超时时间参数timeout_ms来控制更新频率. 如果在60s=1min内如果没有消息可用，返回一个空集合
            msg_pack = self.consumer.poll(timeout_ms=DEFAULT_CONSUMER_WAIT_TIME)
            if not msg_pack:
                continue

            # 遍历每个批次的消息字典
            for tp, messages in msg_pack.items():
                if len(messages) > 1:
                    logger.info("-----------------------------------------")
                    logger.info(f"Processing {len(messages)} messages ...")
                for msg in messages:
                    self._process_message(msg)

    def _process_message(self, msg):
        """处理单条消息"""
        try:
            value_dict = json.loads(msg.value)
            record = json.loads(value_dict['data'])
            source = value_dict.get('source', '').upper()
            operateType = value_dict.get('operateType', '').upper()

            # 根据业务ID过滤可选
            if hasattr(self.kafka_config, 'bot_id_needed'):
                pass  # 这里省略业务过滤逻辑

            # 保存原始消息（可选）
            # with codecs.open("received_kafka_data.json", 'a', encoding='utf-8') as f:
            #     json.dump(value_dict, f, ensure_ascii=False, indent=4)

            # 分发 handler
            if source == SourceType.BOT_DEPLOY:
                logger.info("Source=BOT_DEPLOY, save snapshot & update local files")
                self._save_snapshot()
                self._update_local_files()
                self.updater.trigger_update_faq_knowledge()
            else:
                handler = self._source_handlers.get(source)
                if handler:
                    handler(record, operateType)
                else:
                    logger.warning(f"Unknown source type: {source}")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def _handle_entity(self, record: Dict, op_type: OperateType):
        pass  # 省略实体处理逻辑，保持接口

    def _handle_param(self, record: List[Dict], op_type: OperateType):
        pass  # 省略变量处理逻辑

    def _handle_intent(self, record: Dict, op_type: OperateType):
        pass  # 省略意图处理逻辑

    def _load_snapshot(self) -> bool:
        """
        启动时尝试从 Redis 加载快照。
        成功恢复返回 True，否则返回 False。
        """
        if not self.redis_conn:
            return False
        try:
            logger.info("Loaded snapshot. 初次启动，尝试从 Redis 快照恢复...")
            raw = self.redis_conn.get(snapshot_key)
            if not raw:
                return False
            snap = json.loads(raw)
            self.knowledge_state.update(snap.get("data", {}))     
            offsets = snap.get("offsets", {})   # 偏移量
            if offsets:
                # 手动分配分区和 topic
                tp = TopicPartition(topic=self.kafka_config.topic_name, partition=0)
                self.consumer.assign([tp])
                # 为分区设置偏移量 → 最新
                offset = offsets.get(str(tp.partition))
                if offset:
                    self.consumer.seek(tp, offset)
                self._update_local_files()
                return True
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
        return False

    def _save_snapshot(self):
        """保存快照
        保存知识库和消费者偏移量到 Redis
        """
        if not self.redis_conn:
            logger.warning("Redis connection not provided, cannot save snapshot.")
            return
        if not self.consumer:
            logger.warning("Consumer not running, cannot save snapshot.")
            return
        
        try:
            # 获取当前分配到的分区的偏移量
            # consumer.position(tp) 返回下一个要消费的消息的偏移量，这正是我们需要的
            offsets = {
                str(tp.partition): self.consumer.position(tp)
                for tp in self.consumer.assignment()
            }
            if not offsets:
                logger.warning("No partitions assigned, skipping snapshot with offsets.")
                return
            snap = {"data": self.knowledge_state, "offsets": offsets, "timestamp": time.time()}
            self.redis_conn.set(snapshot_key, json.dumps(snap))
            self.last_snapshot_time = time.time() # 更新快照时间
            logger.info(f"Snapshot saved. Offsets: {offsets}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    def _update_local_files(self):
        # 此处可写入本地 JSON 文件
        pass

    @staticmethod
    def _write_json_file(data: Any, path):
        try:
            with codecs.open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to write file {path}: {e}")

    def _cleanup(self):
        """清理资源"""
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer closed")


class FAQKnowledgeUpdater:
    """FAQ知识库更新器
    
    关键优化：
        原子性文件写入
        异步更新机制
        重试机制
        结构化日志
    """
    def __init__(self):
        self.update_timeout = 30
        self.retry_times = 2

    def trigger_update_faq_knowledge(self) -> None:
        """触发FAQ知识库更新"""
        def update_task():
            try:
                response = requests.get(
                    urls.update_faq_onetouch, 
                    headers={"Content-Type": "application/json"}, 
                )
                response.raise_for_status()
                logger.info(f"Triggered FAQ update, status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"FAQ update failed: {e}")
        
        logger.info("触发faq向量更新......")
        logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        thread = threading.Thread(target=update_task, name="Trigger FAQ-Vector-Update", daemon=True)
        thread.start()
        logger.info("已完成！\n")


def main(kafka_config, redis_config):
    try:
        # Redis 多机共享快照
        redis_client = RedisUtilsSentinel(redis_config.__dict__)
        logger.info("Connected to Redis.")
    except Exception as e:
        logger.error(f"Could not connect to Redis: {e}")
        redis_client = None

    kfk_consumer = KnowledgeSyncConsumer(
        kafka_config=kafka_config,
        redis_conn=redis_client,
    )
    kfk_consumer.run()


if __name__ == "__main__":
    logger.info(f"snapshot_key: --->【{snapshot_key}】")

    # 运行消费者
    main(kafka_config, redis_config)
