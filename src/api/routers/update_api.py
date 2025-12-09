import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings
warnings.filterwarnings("ignore")

import re
import time
from datetime import datetime
from dataclasses import dataclass
import importlib
import traceback
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import json
import requests
import torch
import gc
import logging
import pandas as pd

from fastapi import FastAPI, BackgroundTasks
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi import Depends, File, Body, File, Form, Query
from fastapi.responses import JSONResponse, Response
import uvicorn
import asyncio
# from pydantic import BaseModel, validator
# from starlette.requests import Request

from configs.config import paths
from schemas.request.faq_schema import *
from utils import logger
import application

"""
FAQ系统更新机制，包含四种不同的更新方式：

full_update: 全量更新整个FAQ系统，包括重新加载所有索引
incremental_update: 增量更新，只添加新文档而不重新构建整个索引

2025.10.10
基于 FastAPI.BackgroundTasks 实现异步任务调度，将 I/O 密集型的向量索引重建操作移至后台执行，
消除同步更新引发的服务阻塞与性能瓶颈，提升系统可用性与响应效率。
"""

router = APIRouter()

# 从主应用获取模块实例
faq_sys = application.get_main_faq()
uie = application.get_uie()

# 全局变量声明，跟踪增量添加的文档
global_new_docs: List[Dict] = []

# 创建Redis客户端
from utils import RedisUtilsSentinel
from configs.config import redis_config, snapshot_key
redis_client = RedisUtilsSentinel(redis_config.__dict__)


def torch_gc():
    """释放PyTorch的GPU缓存"""
    try:
        # 检查是否有可用的CUDA设备
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片
        print(f"gpu内存清理完成！")
        # gc.collect()
        # print(f"垃圾回收完成！")
    except Exception as e:
        print(f"清理内存时出错: {e}")


@router.get("/full_update")
def full_update(background_tasks: BackgroundTasks):
    """全量更新 FAQ 系统
    更新全局变量、重新加载Faiss和BM25索引
    """
    try:
        logger.info("开始全量更新 FAQ 系统...")
        
        global faq_sys, uie
        global global_new_docs
        global_new_docs = []

        # 重新创建faq模块实例
        from module import recall_v2
        importlib.reload(recall_v2)
        _qa_paths = [paths.qa.robot, paths.kafka.onetouch, paths.qa.greeting]
        _docs = faq_sys._get_docs(_qa_paths)
        # 将大规模向量索引重建操作移至后台执行
        def update_large_vectors(_docs):
            faq_sys.recall_module = recall_v2.Recall(_docs) 
            logger.info(f"索引统计: {faq_sys.recall_module.faiss.get_stats()}")
        background_tasks.add_task(update_large_vectors, _docs)
        # return {"message": "Processing started"}

        # 重新加载实体抽取器
        from src.components.extractors.ner import EntityExtractor
        _uie = EntityExtractor(paths.slot_data, paths.entity_data)
        application.set_uie(_uie)

        # 更新意图与实体字典

        torch_gc()  # 随着更新次数增多，显存占用会变大，所以顶一个 torch_gc() 方法完成对显存的回收

        logger.info("FAQ 系统全量更新成功")
        return {'status': True, 'msg': 'Successfully updated all FAQ components'}

    except Exception as e:
        error_msg = f"[full_update] 更新失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


@router.get("/incremental_update")
def incremental_update(request: Request):
    """
    增量更新 FAQ 知识库
    """
    try:
        logger.info("开始 FAQ 知识增量更新")

        # 获取新文档
        new_docs = faq_sys._get_docs([paths.qa.increment])
        if not new_docs:
            logger.info("未检测到新增文档，跳过更新")
            return {'status': True, 'msg': 'No new documents found to update.'}
        
        # 执行核心更新逻辑
        faq_sys.recall_module.faiss.add_documents(new_docs)
        logger.info(f"索引统计: {faq_sys.recall_module.faiss.get_stats()}")

        # 更新全局变量
        global global_new_docs
        global_new_docs.extend(new_docs)

        torch_gc()
        
        logger.info("FAQ 知识增量更新成功")
        return {'status': True, 'msg': f'Successfully updated {len(new_docs)} entries'}

    except Exception as e:
        error_msg = f"[incremental_update] 更新失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


if __name__ == "__main__":
    uvicorn.run(
        app="faq_service:router",   # 注意替换为实际运行的模块路径
        host="0.0.0.0",
        port=8091,
        workers=1,
        reload=True,
    )