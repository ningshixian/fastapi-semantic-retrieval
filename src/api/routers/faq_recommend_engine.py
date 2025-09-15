import os
import warnings
warnings.filterwarnings("ignore")

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import json
import requests
import traceback

from contextlib import redirect_stdout, redirect_stderr
with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
    import some_text_cleaning_lib as text_cleaner   # 泛化文本处理库
    import pandas as pd

from fastapi import FastAPI, APIRouter, Request
from fastapi import Depends, File, Body, Form, Query
from fastapi.responses import JSONResponse
import uvicorn

# ========== 工具 & 配置（业务隐去，保持通用结构） ==========
from utils import logger, log_filter
from utils import data_cleaning
from configs.config import *
from schemas.request.faq_schema import *
from schemas.response import response_code

# FAQ 系统初始化（业务具体实现隐去）
from src.main import app as application
faq_sys_rec = application.get_faq_sys_rec()

router = APIRouter()

@dataclass
class SearchResult:
    """搜索结果数据类"""
    text: str
    origin_text: str
    rec_time: int
    response4dm: str
    detail_results: Union[str, List[Dict[str, Any]]]


class TextProcessor:
    """文本预处理"""
    @staticmethod
    def standardize(query: str) -> str:
        if not query:
            return ""
        query = data_cleaning.clean_text(query)
        # 高级文本处理
        query = text_cleaner.clean_text(
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


class FAQRecommendEngine:
    """FAQ推荐搜索引擎"""
    
    def __init__(self, faq_sys_rec, thredshold=0.74):
        self.faq_sys_rec = faq_sys_rec
        self.thredshold = thredshold
        self.text_processor = TextProcessor()
    
    def conversation_rewriter(self, query, history):
        """对话改写"""
        payload = json.dumps({
            "query": query, 
            "session_id": "rec-session",
            "user_history": history,
        })
        try:
            response = requests.post(
                SOME_SUMMARY_SERVICE_URL, 
                headers={"Content-Type": "application/json"}, 
                data=payload, 
                timeout=10
            )
            return response.json().get("data", {}).get("conversationRewriter", [query])[0]
        except Exception as e:
            traceback.print_exc()
            return query
    
    def customized_search(self, 
                          text: Union[str, List[str]],
                          car_type: str = 'N/A', 
                          top_k: int = 5,
                          history: List = [], 
                          search_strategy: str = 'hybrid'
                          ) -> SearchResult:
        response = SearchResult(
            text=text,
            origin_text=text,
            rec_time=int(time.time()),  
            response4dm="搜索完成", 
            detail_results=[]
        )
        
        validation_error = self._validate_input(text, search_strategy, top_k)
        if validation_error:
            response.response4dm = validation_error
            return response
        
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            response.response4dm = "输入格式错误"
            return response
        
        text = [self.text_processor.standardize(t) for t in text]
        
        if not text or not all(text):
            response.response4dm = "无法处理的无效输入"
            return response
        
        search_results = self._execute_search(text, top_k, search_strategy)

        # 低置信度二次改写
        if search_results and search_results[0].get("score", 0) < self.thredshold:
            history = [
                {"round": len(history)-idx, "query": hist['content']}
                for idx, hist in enumerate(reversed(history))
                if hist.get("direction") == 1
            ]
            if history:
                rewrite_text = self.conversation_rewriter(response.origin_text, history)
                logger.info(f"调用改写接口, 改写结果: {rewrite_text}")
                search_results = self._execute_search([rewrite_text], top_k, search_strategy)
                response.text = rewrite_text
            if not search_results or search_results[0].get("score", 0) < self.thredshold:
                search_results = []
        
        if not search_results:
            response.response4dm = "未匹配到答案"
            return response
        
        search_results = [r for r in search_results if r.get("score", 0) > self.thredshold][:top_k]
        
        response.detail_results = [
            {
                "question_id": r.get("meta", {}).get("question_id", ""),
                "standard_sentence": r.get("meta", {}).get("main_question", ""),
                "match_sentence": r.get("content", ""),
                "answer": (
                    r.get("meta", {}).get("answers", [{}])[0].get("answer_content", "")
                ),
                "score": round(r.get("score", 0), 4),
                "source": "知识库",
                "car_type": ""
            }
            for r in search_results
        ]
        return response
    
    def _validate_input(self, text: Any, search_strategy: str, top_k: int) -> Optional[str]:
        if not text:
            return "输入不能为空"
        if search_strategy not in ['hybrid', 'bm25', 'embedding']:
            return "检索策略不支持"
        if not isinstance(top_k, int) or top_k <= 0:
            return "top_k必须是正整数"
        return None
    
    def _execute_search(self, text: List[str], top_k: int, search_strategy: str) -> List[Dict]:
        results = self.faq_sys_rec.search(text, size=top_k, search_strategy=search_strategy)
        return results


@router.post("/recommend", summary="FAQ推荐", name="FAQ推荐")
@log_filter
def recommend_api(request: Request, item: Item4rec):
    engine = FAQRecommendEngine(faq_sys_rec, thredshold=float(DEFAULT_THRESHOLD))
    result = engine.customized_search(
        text=item.text,
        top_k=item.topN,
        history=item.message, 
        search_strategy=item.search_strategy,
        car_type=item.carType, 
    )
    return response_code.resp_200(data=result.__dict__)


if __name__ == "__main__":
    uvicorn.run(
        app="faq_recommend_engine:router",
        host="0.0.0.0",
        port=8091,
        workers=1,
        reload=True,
    )