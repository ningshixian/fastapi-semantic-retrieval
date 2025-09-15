from typing import List
from fastapi import APIRouter, Body, Request

from schemas.response import response_code
from src.main import app as application

router = APIRouter()


# 这里模拟一个实体抽取函数（实际需要替换成你自己的模型或算法）
def entity_extract_from_text(text, entity_mode: str = "default"):
    """
    通用实体抽取函数
    """
    flag, entity_value, start_idx, end_idx = application.get_uie().entity_extract(
        text, entity_mode, entity={'entityName':''}
    )
    return flag, entity_value, start_idx, end_idx


@router.post("/entity_extract", summary="实体抽取", name="实体抽取")
def entity_extract(
    request: Request,
    text: str = Body(..., embed=True)
):
    """
    通用实体抽取接口
    """
    # 保证输入是列表形式
    if isinstance(text, str):
        text = [text]
    
    # 调用通用实体抽取方法
    flag, entity_value, start_idx, end_idx = entity_extract_from_text(
        text, entity_mode="default"
    )

    match_entity: List[dict] = []
    if flag:
        match_entity.append({
            'start': start_idx,            # 实体开始位置
            'end': end_idx,                # 实体结束位置
            'entity_name': "",              # 实体名称，可为空
            'entity_value': entity_value,   # 实体值
            'entity_mode': "defaultEntity", # 实体类型，更通用的命名
            'param_name': "",               # 参数名
            "prebuild": False,              # 是否系统预置实体
            "is_prebuild": False            # 是否系统预置实体
        })

    return response_code.resp_200(data=match_entity)