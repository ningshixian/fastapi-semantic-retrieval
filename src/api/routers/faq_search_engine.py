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
    """搜索结果数据类"""
    text: str
    origin_text: str
    response4dm: str
    match_type: int
    confidence: float
    threshold: Dict[str, float]
    detail_results: List[Dict[str, Any]]


class QueryProcessor:
    """文本处理类"""
    
    @staticmethod
    def standardize(query: str) -> str:
        if not query:
            return ""
        query = data_cleaning.clean_text(query)
        query = text_cleaning_lib.clean_text(
            text=query,
            remove_html_tag=True,
            convert_full2half=True,
            remove_exception_char=False,
            remove_url=True,
            remove_email=True,
            remove_redundant_char=True,
            remove_parentheses=False,
            remove_phone_number=False,
        )
        return query.strip()


class ConditionMatcher:
    """条件匹配类（示例逻辑，具体业务需自行扩展）"""
    
    @staticmethod
    def match_conditions(answer: Dict[str, Any], 
                        user_conditions: Dict[str, Any]) -> float:
        score = 0.0
        score += ConditionMatcher._match_car_type(answer, user_conditions)
        score += ConditionMatcher._match_ota_version(answer, user_conditions)
        if not ConditionMatcher._validate_answer(answer):
            score = 0
        return round(score, 3)
    
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
                return 0.4
        else:
            if not car_label:
                return 0.4
        return 0.0
    
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
                        return 0.3
            except ValueError:
                logger.warning(f"版本转换失败: {lowest_v}, {highest_v}")
        else:
            if not lowest_v and not highest_v:
                return 0.3
        return 0.0
    
    @staticmethod
    def _validate_answer(answer: Dict[str, Any]) -> bool:
        """有效性验证"""
        if answer.get("status") != 1:
            return False
        current_time = str(datetime.now())
        if answer.get("valid_begin_time") and answer["valid_begin_time"] > current_time:
            return False
        if answer.get("valid_end_time") and answer["valid_end_time"] < current_time:
            return False
        return True

    @staticmethod
    def _is_valid_time(valid_begin_time, valid_end_time, current_time):
        if valid_begin_time:
            try:
                if datetime.strptime(valid_begin_time, "%Y-%m-%d %H:%M:%S") > current_time:
                    return False
            except ValueError:
                return False
        if valid_end_time:
            try:
                if datetime.strptime(valid_end_time, "%Y-%m-%d %H:%M:%S") < current_time:
                    return False
            except ValueError:
                return False
        return True

    @staticmethod
    def validate_results(search_results):
        """过滤无效结果"""
        current_time = datetime.now()
        valid_results = []
        for result in search_results:
            meta = result.get('meta', {})
            if meta.get('status') != 1 or meta.get('is_delete', 0) != 0:
                continue
            if not ConditionMatcher._is_valid_time(
                meta.get('valid_begin_time'), 
                meta.get('valid_end_time'), 
                current_time
            ):
                continue
            answer_content_list = meta.get('answer_content_list', [])
            valid_answers = []
            for answer in answer_content_list:
                if answer.get('status') != 1 or answer.get('is_delete', 0) != 0:
                    continue
                if ConditionMatcher._is_valid_time(
                    answer.get('valid_begin_time'),
                    answer.get('valid_end_time'),
                    current_time
                ):
                    valid_answers.append(answer)
            meta['answer_content_list'] = valid_answers
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
    """FAQ搜索引擎主类（保留通用搜索流程）"""
    
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

        search_results = self._execute_search(
            text, top_k, search_strategy, exclude_team
        )
        search_results = ConditionMatcher.validate_results(search_results)
        if not search_results:
            response.response4dm = "未匹配到结果"
            return response
        search_results = self._apply_priority_logic(search_results)
        if not search_results:
            response.response4dm = "未匹配到结果"
            return response
        search_results = search_results[:top_k]
        if self.intent_extractor:
            for res in search_results:
                res['match_entity'] = self.intent_extractor.extract_entities(text, res)

        response.text = text[0]
        response.response4dm = self._build_response4dm(search_results)
        response.match_type = self._classify_match_type(search_results[0]['score'])
        response.confidence = round(search_results[0]['score'], 4)
        response.detail_results = search_results
        return response
    
    def _execute_search(self, text, top_k, search_strategy, exclude_team):
        results = self.faq_sys.search(text, size=top_k, search_strategy=search_strategy)
        for result in results:
            result["meta"]['source'] = self.lib_name2id.get(result["meta"]["source"], -1)
        if exclude_team:
            results = [r for r in results if r["meta"]["source"] not in exclude_team]
        return results
    
    def _apply_priority_logic(self, results):
        high_priority_results = [r for r in results if r["meta"].get("source") in self.config.high_priority]
        low_priority_results = [r for r in results if r["meta"].get("source") in self.config.low_priority]
        top_result = results[0]
        SCORE_DIFF_THRESHOLD = 0.1
        should_use_low_priority = (
            top_result["meta"]["source"] in self.config.low_priority
            and top_result["score"] > self.config.low_threshold
            and (not high_priority_results or 
                 top_result["score"] - high_priority_results[0]["score"] > SCORE_DIFF_THRESHOLD)
        )
        return low_priority_results if should_use_low_priority else high_priority_results

    def _classify_match_type(self, score: float) -> int:
        if score >= self.config.high_threshold:
            return 1
        elif score >= self.config.low_threshold:
            return 3
        else:
            return 0
    
    def _build_response4dm(self, results: List[Dict]) -> str:
        lines = [f"{x['score']:.4f}\t{x['content']}" for x in results]
        message = "\n".join(lines)
        if results:
            message += "\n\nTop1：" + results[0].get("content", "")
        return message


def create_faq_engine(faq_sys, intent_dict, param_dict, entity_dict):
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
    engine = create_faq_engine(faq_sys, intent_dict, param_dict, entity_dict)
    response = engine.customized_search(
        text=item.text,
        user_id=item.user_id,
        top_k=item.top_k,
        search_strategy=item.search_strategy,
        exclude_team=item.exclude_team,
    )
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