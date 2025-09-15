import json
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.encoders import jsonable_encoder

# 模拟引入通用响应工具（敏感路径已泛化）
from utils import response as response_util  
# 模拟引入应用实例（业务细节已打码）
from app.main import app_instance  

router = APIRouter()

@router.post("/similarity", summary="相似度计算", name="相似度计算")
def similarity(
    request: Request, 
    text1: str = Body(..., embed=True),
    text2: str = Body(..., embed=True)
):
    """
    计算两个文本之间的向量相似度
    """
    if not text1 or not text2:
        return response_util.resp_400(data="输入不能为空")
    
    # 调用通用相似度计算模块（敏感调用已泛化）
    score = app_instance.get().similarity_module.calculate_similarity(text1, text2)
    
    # 保证输出为字典格式，并避免直接返回浮点类型
    return response_util.resp_200(data={'cosine_similarity': round(score[0], 6)})


# 示例：批量相似度计算接口（已注释，可按需使用）
# @router.post("/batchsimilarity")
# def batchsimilarity(queries: List[str] = Body(...), texts: List[str] = Body(...)):
#     """
#     批量计算多个查询与多个文本之间的相似度。
#     返回每个查询对应的按相似度排序的结果列表。
#     """
#     return app_instance.get().similarity_module.batch_calculate(queries, texts)
