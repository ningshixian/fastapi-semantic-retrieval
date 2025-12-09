import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")

import re
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import json
import logging

from contextlib import redirect_stdout, redirect_stderr
with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
    import some_text_cleaning_lib as text_cleaning_lib
    import pandas as pd

from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import JSONResponse
import uvicorn

# 从工具模块引入通用方法
from utils import logger, log_filter, data_cleaning
from configs.config import *
from schemas.request.some_schema import *
from schemas.response import response_code
from src.main import app as application

"""
启动方式示例：
CUDA_VISIBLE_DEVICES=0 python api.py
gunicorn api:router --bind=0.0.0.0:8091 --workers=1 -k uvicorn.workers.UvicornWorker
"""

router = APIRouter()

# 从缓存服务获取数据（示例：Redis / 其他缓存系统）
from utils import RedisUtilsSentinel
from configs.config import redis_config, snapshot_key
redis_client = RedisUtilsSentinel(redis_config.__dict__)
try:
    snap = json.loads(redis_client.get(snapshot_key))
    intent_dict, param_dict, entity_dict = (
        snap["data"]["intent2param"],
        snap["data"]["param2ent"], 
        snap["data"]["ent2vocab"]
    )
except Exception:
    intent_dict, param_dict, entity_dict = {}, {}, {}

# 系统主实例（此处调用的对象为示例）
faq_sys = application.get()
uie = application.get_uie()


@dataclass
class SearchResult:
    """搜索结果数据类
    用于封装FAQ搜索的完整结果信息
    """
    text: str           # 处理后的查询文本
    origin_text: str    # 原始查询文本
    response4dm: str    # 传给DM用于前端展示的响应内容
    match_type: int     # 匹配类型（0：空 1：精准匹配 3：模糊匹配）
    confidence: float   # 置信度（top1结果的得分）
    threshold: Dict[str, float]
    detail_results: List[Dict[str, Any]]    # 详细的检索结果列表


class QueryProcessor:
    """文本处理类

    主要功能：
    1. 基础文本清洗（使用data_cleaning工具）
    2. 高级文本清洗（使用jionlp工具）
    """
    
    @staticmethod
    def standardize(query: str) -> str:
        if not query:
            return ""
        
        # 基础文本清洗
        query = data_cleaning.clean_text(query)
        
        # 高级文本清洗
        # 补充：去除文本中的异常字符、冗余字符、HTML标签、括号信息、URL、E-mail、电话号码，全角字母数字转换为半角
        query = jionlp.clean_text(
            text=query,
            remove_html_tag=True,   # HTML标签
            convert_full2half=True, # 全角字母数字转换为半角
            remove_exception_char=False, 
            # remove_exception_char=True, # 删除文本中异常字符，主要保留汉字、常用的标点，单位计算符号，字母数字等
            remove_url=True,
            remove_email=True, 
            remove_redundant_char=True, # 删除文本中冗余重复字符
            remove_parentheses=False,    # 删除括号内容 ✖
            remove_phone_number=False,
        )
        return query.strip()


class ConditionMatcher:
    """条件匹配类（示例逻辑，具体业务需自行扩展）"""
    
    @staticmethod
    def _match_car_type(answer: Dict[str, Any], 
                       user_conditions: Dict[str, Any]) -> float:
        """车型匹配逻辑示例"""
        car_label = " / ".join([
            x.get("series_name", "") + " " + x.get("model_name", "") 
            for x in answer.get("car_label_list", [])
        ])
        user_cars = user_conditions.get("car_type", [])
        normalize = lambda x: re.sub(r'[_\s]', '', x.lower())
        if user_cars:
            if any([normalize(car) in normalize(car_label) for car in user_cars]):
                return True
        else:
            if not car_label:
                return True
        return False
    
    @staticmethod
    def _match_ota_version(answer: Dict[str, Any], 
                          user_conditions: Dict[str, Any]) -> float:
        """版本匹配逻辑示例"""
        lowest_v = answer.get("lowest_version", "")
        highest_v = answer.get("highest_version", "")
        user_versions = user_conditions.get("version", [])
        
        if user_versions:
            try:
                if lowest_v and highest_v:
                    if any([float(lowest_v) <= float(ver) <= float(highest_v) 
                           for ver in user_versions]):
                        return True
            except ValueError:
                logger.warning(f"版本转换失败: {lowest_v}, {highest_v}")
        else:
            if not lowest_v and not highest_v:
                return True
        return False

    @staticmethod
    def _is_valid_time(valid_begin_time, valid_end_time, current_time):
        """验证时间是否有效
        """
        # 检查开始时间
        if valid_begin_time:
            try:
                if datetime.strptime(valid_begin_time, "%Y-%m-%d %H:%M:%S") > current_time:
                    return False
            except ValueError:
                return False
        # 检查结束时间
        if valid_end_time:
            try:
                if datetime.strptime(valid_end_time, "%Y-%m-%d %H:%M:%S") < current_time:
                    return False
            except ValueError:
                return False
        return True

    @staticmethod
    def validate_results(search_results):
        """筛选最终结果作为 Response 的 detail_results!
        - 主问题、答案的启用状态 (status=1)
        - 主问题、答案的有效期判断
        - 主问题、答案的删除状态 (is_delete=0)
        """
        current_time = datetime.now()
        valid_results = []
        for result in search_results:
            meta = result.get('meta', {})
            # 检查主问题的启用状态、删除状态、有效期
            if meta.get('status') != 1 or meta.get('is_delete', 0) != 0:
                continue
            if not ConditionMatcher._is_valid_time(
                meta.get('valid_begin_time'), 
                meta.get('valid_end_time'), 
                current_time
            ):
                continue
            # 检查答案的启用状态、删除状态、有效期
            answers = meta.get('answers', [])
            valid_answers = []
            for answer in answers:
                if answer.get('status') != 1 or answer.get('is_delete', 0) != 0:
                    continue
                if not ConditionMatcher._is_valid_time(
                    answer.get('valid_begin_time'),
                    answer.get('valid_end_time'),
                    current_time
                ):
                    continue
                # # 检查答案车型是否匹配
                # if not ConditionMatcher._match_car_type(answer, user_conditions):
                #     continue
                # # 检查答案OTA版本是否匹配
                # if not ConditionMatcher._match_ota_version(answer, user_conditions):
                #     continue
                valid_answers.append(answer)
            
            # 更新答案列表
            meta['answers'] = valid_answers
            valid_results.append(result)
        return valid_results


class IntentEntityExtractor:
    """意图-实体提取类（泛化示例，不含具体业务逻辑）"""

    def __init__(self, intent_dict, param_dict, entity_dict, low_threshold):
        self.intent_dict = intent_dict
        self.param_dict = param_dict
        self.entity_dict = entity_dict
        self.low_threshold = low_threshold
        self.entity_id2name = {
            "0": ("prebuildEntity", "系统实体"), 
            "1": ("enumerateEntity", "枚举实体"), 
            #...
        }
        self.intentname2id = {}
        self._build_intent_name_map()
    
    def _build_intent_name_map(self):
        for intent_id, intent_info in self.intent_dict.items():
            self.intentname2id[intent_info.get('intentName', '')] = intent_id
            self.intentname2id[intent_info.get('primaryName', '')] = intent_id
    
    def extract_entities(self, text: Union[str, List[str]], 
                        result: Dict[str, Any]) -> List[Dict[str, Any]]:
        entities = []
        if result["meta"].get('source') != 2 or result['score'] < self.low_threshold:
            return entities
        intent_id = self.intentname2id.get(result['content'])
        if not intent_id:
            return entities
        intent = self.intent_dict.get(intent_id, {})
        for param_id in intent.get('paramIdList', []):
            param = self.param_dict.get(param_id, {})
            entity_id = param.get('entityId', '')
            entity = self.entity_dict.get(entity_id, {})
            if not entity:
                continue
            entity_type = str(entity.get('entityType', ''))
            entity_mode = self.entity_id2name.get(entity_type, [''])[0]
            flag, entity_value, start_idx, end_idx = uie.entity_extract(
                text, entity_mode, entity
            )
            if flag:
                entities.append({
                    'start': start_idx,
                    'end': end_idx,
                    'entity_name': entity.get('entityName', ''),
                    'entity_value': entity_value,
                    'entity_mode': entity_mode,
                    'param_name': param.get('paramName', ''),
                    'prebuild': False,
                })
        return entities


class FAQSearchEngine:
    """FAQ搜索引擎主类
    
    主要功能：
    1. 输入验证和预处理
    2. 调用FAQ系统执行搜索
    3. 应用优先级逻辑筛选结果
    4. 处理意图和实体提取
    5. 构建最终响应
    """
    
    def __init__(self, faq_sys, config):
        self.faq_sys = faq_sys
        self.config = config
        self.query_processor = QueryProcessor()
        self.intent_extractor = None
        self.lib_name2id = {
            "knowledge_base": 1,
            "intent_lib": 2,
            "custom_smalltalk": 4,
            "builtin_smalltalk": 5,
        }
    
    def set_intent_extractor(self, intent_dict, param_dict, entity_dict):
        self.intent_extractor = IntentEntityExtractor(
            intent_dict, param_dict, entity_dict, self.config.low_threshold
        )
    
    def customized_search(self, 
               text: Union[str, List[str]],
               user_id: str = 'test',
               top_k: int = 5,
               search_strategy: str = 'hybrid',
               exclude_team: List[int] = None) -> SearchResult:

        response = SearchResult(
            text=text,
            origin_text=text,
            response4dm="",
            match_type=-1,
            confidence=-1,
            threshold={"high": self.config.high_threshold, "low": self.config.low_threshold},
            detail_results=[]
        )

        if isinstance(text, str):
            text = [text]
        text = [self.query_processor.standardize(t) for t in text]
        if not text or not all(text):
            response.response4dm = "无效输入"
            return response

        # 1、执行搜索
        search_results = self._execute_search(
            text, top_k, search_strategy, exclude_team
        )
        # 2、复杂filters逻辑 → 筛选答案
        search_results = ConditionMatcher.validate_results(search_results)
        if not search_results:
            response.response4dm = "未匹配到结果"
            return response
        # 3、应用匹配优先级逻辑
        search_results = self._apply_priority_logic(search_results)
        if not search_results:
            response.response4dm = "未匹配到结果"
            return response
        # 保留top_k
        search_results = search_results[:top_k]
        # 4、处理意图和实体
        if self.intent_extractor:
            for res in search_results:
                res['match_entity'] = self.intent_extractor.extract_entities(text, res)
        # 5、构建响应
        response.text = text[0]
        response.response4dm = self._build_response4dm(search_results)
        response.match_type = self._classify_match_type(search_results[0]['score'])
        response.confidence = round(search_results[0]['score'], 4)
        response.detail_results = search_results
        return response
    
    def _execute_search(self, text, top_k, search_strategy, exclude_team):
        # 执行搜索
        results = self.faq_sys.search(text, size=top_k, search_strategy=search_strategy)
        # 转换库名到ID
        for result in results:
            result["meta"]['source'] = self.lib_name2id.get(result["meta"]["source"], -1)
        # 排除指定库
        if exclude_team:
            results = [r for r in results if r["meta"]["source"] not in exclude_team]
        return results
    
    def _apply_priority_logic(self, results):
        """应用匹配优先级逻辑"""
        # 按来源分组结果
        high_priority_results = [r for r in results if r["meta"].get("source") in self.config.high_priority]
        low_priority_results = [r for r in results if r["meta"].get("source") in self.config.low_priority]
        # 获取最高分的结果
        top_result = results[0]
        # 判断是否应该使用低优先级（寒暄库）
        SCORE_DIFF_THRESHOLD = 0.1
        should_use_low_priority = (
            # top结果来自低优先级库
            top_result["meta"]["source"] in self.config.low_priority
            # 分数超过阈值
            and top_result["score"] > self.config.low_threshold
            # 与高优先级最高分相差超过阈值
            and (not high_priority_results or 
                 top_result["score"] - high_priority_results[0]["score"] > SCORE_DIFF_THRESHOLD)
        )
        return low_priority_results if should_use_low_priority else high_priority_results

    def _classify_match_type(self, score: float) -> int:
        if score >= self.config.high_threshold:
            return 1  # 精准匹配
        elif score >= self.config.low_threshold:
            return 3  # 模糊匹配
        else:
            return 0  # 不回复
    
    def _build_response4dm(self, results: List[Dict]) -> str:
        lines = [f"{x['score']:.4f}\t{x['content']}" for x in results]
        message = "\n".join(lines)
        if results:
            message += "\n\nTop1：" + results[0].get("content", "")
        return message


def create_faq_engine(faq_sys, intent_dict, param_dict, entity_dict):
    # 创建FAQ搜索引擎实例
    engine = FAQSearchEngine(faq_sys, threshold_priority_config)
    engine.set_intent_extractor(intent_dict, param_dict, entity_dict)
    return engine


@router.get("/")
async def read_root():
    return response_code.resp_200(data={"message": "Welcome"})


@router.post("/predict", summary="FAQ检索", name="FAQ检索")
@log_filter
def predict(
    request: Request, 
    item: Item4cc
):
    # 创建搜索引擎（可以在应用启动时创建并复用）
    engine = create_faq_engine(faq_sys, intent_dict, param_dict, entity_dict)
    # 执行搜索
    response = engine.customized_search(
        text=item.text,
        user_id=item.user_id,
        top_k=item.top_k,
        search_strategy=item.search_strategy,
        exclude_team=item.exclude_team,
    )
    # 转换为响应格式
    response_data = response.__dict__
    if response.match_type == -1:
        return response_code.resp_4001(data=response_data)
    else:
        return response_code.resp_200(data=response_data)


if __name__ == "__main__":
    uvicorn.run(
        app="faq_search_engine:router",
        host="0.0.0.0",
        port=8091,
        workers=1,
        reload=True,
    )