#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Hui
# @Desc: { kafka连接处理模块 }
# @Date: 2023/05/03 21:14
import json
import time
# 导入 kafka-python 的相关模块
from kafka import KafkaProducer, KafkaConsumer
from kafka.structs import TopicPartition
from kafka.errors import KafkaError


class kafka_service:
    def __init__(self, kafka_ip_list, topics):
        self.kafka_ip_list = kafka_ip_list
        self.topics = topics    # []

        self.consumer = self._consumer(topics)

    def _producer(self, msg_dict, topic):
        producer = KafkaProducer(
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,  # 重试次数
            bootstrap_servers=self.kafka_ip_list,
            # # 传输时的压缩格式
            # compression_type="gzip",
        )
        producer.send(topic, value=msg_dict, partition=0)   # 发送到指定的消息主题（异步，不阻塞）
        producer.close()

    def _consumer(self):
        consumer = KafkaConsumer(
            bootstrap_servers=self.kafka_ip_list,
            client_id="...",
            group_id="...",
            auto_offset_reset="latest",
            enable_auto_commit=False,  # 手动管理偏移量，所以关闭自动提交
            value_deserializer=lambda v: v.decode("utf-8", "ignore"),
            key_deserializer=lambda k: k.decode("utf-8", "ignore"),
            max_poll_records=1,
            session_timeout_ms=120000,
            request_timeout_ms=120001,
        )
        consumer.subscribe(self.topics)  # 订阅指定的主题
        # # 订阅多个 topic
        # consumer.subscribe(pattern='topic1, topic2, topic3')

        # # 获取主题的分区信息
        # logger.info(self.consumer.partitions_for_topic(self.topic))  # {0}
        # # # 获取主题列表
        # # logger.info(self.consumer.topics())
        # # 获取当前消费者订阅的主题
        # logger.info(self.consumer.subscription())
        # # 获取当前消费者topic、分区信息
        # logger.info(self.consumer.assignment())
        # # 获取当前主题{0}的最新偏移量
        # tp = TopicPartition(topic=self.topic, partition=0)
        # offset = self.consumer.position(tp)  # <class 'int'>
        # logger.info(f"分区0 for topic '{self.topic}' 的最新偏移量: {offset}")
        # # ps：【未指定分区】+【消息未指定key】-> 随机地发送到 topic 内的所有可用分区{0} -> 可保证消费消息的顺序性
        return consumer

    def _consume_messages(self):
        while True:
            # 使用 poll 获取批量消息
            # 提供超时时间参数timeout_ms来控制更新频率. 如果在60s=1min内如果没有消息可用，返回一个空集合
            msg_pack = self.consumer.poll(timeout_ms=60*1000)
            if not msg_pack:
                continue

            # 遍历每个批次的消息字典
            for tp, messages in msg_pack.items():
                for msg in messages:
                    self._process_message(msg)

    def _process_message(self, msg):
        """处理单条消息"""
        try:
            value_dict = json.loads(msg.value)
            ...
        except Exception as e:
            print(f"Error processing message: {e}", exc_info=True)

    def _cleanup(self):
        """清理资源"""
        if self.consumer:
            self.consumer.close()
            print("Consumer closed")

