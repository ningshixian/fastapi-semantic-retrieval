import json
from typing import Union, List
from fastapi import APIRouter, Body, Request

# 假设有统一的响应封装方法
from some_module.response import resp_200  

# 假设 app 对象提供搜索和向量处理功能
from some_module.app import app_instance  

router = APIRouter()


@router.post("/embedding_search")
def embedding_search(
    request: Request,
    text: Union[str, List[str]] = Body(..., embed=True),
    topk: int = Body(default=5, embed=True)
):
    """
    向量搜索接口

    Args:
        text: 搜索文本，可以是单个字符串或字符串列表
        topk: 搜索结果数量
    """
    search_results = app_instance.get().search(
        text, size=topk, search_strategy='embedding'
    )
    return resp_200(data=search_results[:topk])


@router.post("/embedding_transform", summary="转换向量", name="转换向量")
def embedding_transform(
    request: Request,
    text: Union[str, List[str]] = Body(..., embed=True)
):
    """
    文本转向量
    """
    if isinstance(text, str):
        text = [text]

    vectors = app_instance.get().vector_service.encode(text)

    # 转成 Python 列表再序列化为 JSON，避免 NumPy 数组无法直接序列化问题
    return resp_200(data={'embedding': json.dumps(vectors.tolist())})