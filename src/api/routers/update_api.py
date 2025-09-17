import os
import warnings
import re
import time
import importlib
import traceback
import json
import torch
import gc
import logging
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

# ========== 配置与工具 ==========
warnings.filterwarnings("ignore")

from utils import logger  # 建议替换为你自己的日志封装
from configs.config import paths   # 各类路径，从配置文件加载
from schemas.response import response_code
from src.main import app as application  # 主应用引入

# 创建Redis客户端
from utils import RedisUtilsSentinel
from configs.config import redis_config, snapshot_key
redis_client = RedisUtilsSentinel(redis_config.__dict__)

# 全局变量
global_new_docs: List[Dict] = []

# 从主应用获取模块实例
faq_sys = application.get()
faq_sys_rec = application.get_faq_sys_rec()
uie = application.get_uie()

# ========== 工具方法 ==========

def torch_gc():
    """释放 PyTorch 的 GPU 缓存"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("GPU 内存清理完成！")
    except Exception as e:
        print(f"清理内存时出错: {e}")


# ========== 路由定义 ==========
router = APIRouter()

@router.get("/full_update")
def full_update(request: Request):
    """
    全量更新 FAQ 系统，包括重新加载索引与模型
    """
    try:
        logger.info("开始全量更新 FAQ 系统...")
        global faq_sys, uie
        global intent_dict, param_dict, entity_dict
        global global_new_docs
        global_new_docs = []

        # 重新加载 recall 模块
        from module import recall_v2
        importlib.reload(recall_v2)
        _qa_paths = [paths.qa.general, paths.qa.special, paths.qa.greeting]  # 示例路径
        _docs = faq_sys._get_docs(_qa_paths)
        faq_sys.recall_module = recall_v2.Recall(_docs) 
        logger.info(f"索引统计: {faq_sys.recall_module.faiss.get_stats()}")

        # 重新加载实体抽取器
        from src.components.extractors.ner import EntityExtractor
        _uie = EntityExtractor(paths.slot_data, paths.entity_data)
        application.set_uie(_uie)

        # 更新意图与实体字典
        try:
            snap = json.loads(redis_client.get(snapshot_key))
            application.set_intent_params(snap["data"].get("intent2param", {}),
                                        snap["data"].get("param2ent", {}),
                                        snap["data"].get("ent2vocab", {}))
            logger.info("Loaded snapshot from Redis successfully.")
        except Exception:
            logger.error(f"Failed to load snapshot from Redis: {e}")

        torch_gc()
        logger.info("FAQ 系统全量更新成功")
        return {'status': True, 'msg': 'Successfully updated all FAQ components'}

    except Exception as e:
        error_msg = f"[full_update] 更新失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


@router.get("/full_update_recommend")
def full_update_recommend(request: Request):
    """
    全量更新 推荐 API 相关索引
    """
    try:
        logger.info("开始全量更新 Recommend API ...")
        global faq_sys_rec

        from module import recall_v2
        importlib.reload(recall_v2)
        _qa_paths = [paths.qa.recommend]
        _docs = faq_sys_rec._get_docs(_qa_paths)
        faq_sys_rec.recall_module = recall_v2.Recall(_docs)

        torch_gc()
        return {'status': True, 'msg': 'Successfully updated Recommend API'}

    except Exception as e:
        error_msg = f"[full_update_recommend] 更新失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


@router.get("/update_faq_special")
def update_faq_special(request: Request):
    """
    增量更新 FAQ 中特定领域的知识
    """
    try:
        logger.info("开始更新 FAQ 特定知识...")
        global faq_sys, uie
        global intent_dict, param_dict, entity_dict

        # 增量添加新文档
        new_docs = faq_sys._get_docs([paths.qa.increment_special])
        faq_sys.recall_module.faiss.add_documents(new_docs)
        logger.info(f"索引统计: {faq_sys.recall_module.faiss.get_stats()}")

        # 重新加载实体抽取器
        from src.components.extractors.ner import EntityExtractor
        _uie = EntityExtractor(paths.slot_data, paths.entity_data)
        application.set_uie(_uie)

        # 更新意图与实体字典
        try:
            snap = json.loads(redis_client.get(snapshot_key))
            application.set_intent_params(snap["data"].get("intent2param", {}),
                                        snap["data"].get("param2ent", {}),
                                        snap["data"].get("ent2vocab", {}))
            logger.info("Loaded snapshot from Redis successfully.")
        except Exception as e:
            logger.error(f"Failed to load snapshot from Redis: {e}")

        torch_gc()
        logger.info("FAQ 特定领域更新成功")
        return {'status': True, 'msg': 'Successfully updated special domain for FAQ'}

    except Exception as e:
        error_msg = f"[update_faq_special] 更新失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


@router.get("/incremental_update")
def incremental_update(request: Request):
    """
    增量更新 FAQ 知识库
    """
    try:
        logger.info("开始 FAQ 知识增量更新")
        new_docs = faq_sys._get_docs([paths.qa.increment])
        faq_sys.recall_module.faiss.add_documents(new_docs)
        logger.info(f"索引统计: {faq_sys.recall_module.faiss.get_stats()}")

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